"""Provider-independent LLM access: any service, any API key.

Services with an OpenAI-compatible chat API (most of them) or the Anthropic messages API are used by name
or by URL over plain HTTPS, with no vendor package needed. Gemini goes through its own SDK (llm.GeminiLLM).

Settings are taken from, in this order: command-line flags, environment variables (LLM_PROVIDER, LLM_MODEL,
LLM_BASE_URL, LLM_API_KEY, or the provider's own key variable such as OPENAI_API_KEY), the optional
.llm_settings.json written by scripts/setup_llm.py (never contains the key; another file can be named in
LLM_SETTINGS_FILE), and the provider's defaults.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_FILE = REPO_ROOT / ".llm_settings.json"
API_FORMATS = ("openai", "anthropic")

# name -> request format, default API address, environment variables searched for the key.
# The addresses are the providers' public ones; override any of them with --base-url.
PROVIDERS = {
    "gemini":     {"api": "gemini", "base_url": None, "key_envs": ("GEMINI_API_KEY", "GOOGLE_API_KEY")},
    "openai":     {"api": "openai", "base_url": "https://api.openai.com/v1", "key_envs": ("OPENAI_API_KEY",)},
    "anthropic":  {"api": "anthropic", "base_url": "https://api.anthropic.com", "key_envs": ("ANTHROPIC_API_KEY",)},
    "openrouter": {"api": "openai", "base_url": "https://openrouter.ai/api/v1", "key_envs": ("OPENROUTER_API_KEY",)},
    "groq":       {"api": "openai", "base_url": "https://api.groq.com/openai/v1", "key_envs": ("GROQ_API_KEY",)},
    "together":   {"api": "openai", "base_url": "https://api.together.xyz/v1", "key_envs": ("TOGETHER_API_KEY",)},
    "mistral":    {"api": "openai", "base_url": "https://api.mistral.ai/v1", "key_envs": ("MISTRAL_API_KEY",)},
    "deepseek":   {"api": "openai", "base_url": "https://api.deepseek.com/v1", "key_envs": ("DEEPSEEK_API_KEY",)},
    "ollama":     {"api": "openai", "base_url": "http://localhost:11434/v1", "key_envs": ()},      # local, no key
    "lmstudio":   {"api": "openai", "base_url": "http://localhost:1234/v1", "key_envs": ()},       # local, no key
    "custom":     {"api": "openai", "base_url": None, "key_envs": ()},   # any OpenAI-compatible service: --base-url
}
KEY_OPTIONAL = ("ollama", "lmstudio", "custom")


# ------------------------------------------------------------------ settings
@dataclass
class ResolvedLLM:
    provider: str
    api: str
    base_url: str | None
    model: str | None
    api_key: str | None
    key_source: str
    extra: dict = field(default_factory=dict)
    json_mode: bool = False

    def describe(self) -> str:
        key = f"...{self.api_key[-4:]} (from {self.key_source})" if self.api_key else "none"
        return (f"provider {self.provider} ({self.api} format), model {self.model or '(provider default)'}, "
                f"address {self.base_url or '(SDK default)'}, key {key}")


def load_settings_file(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("LLM_SETTINGS_FILE") or SETTINGS_FILE)
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def parse_extra(items) -> dict:
    """['reasoning_effort=low', 'top_p=0.9'] -> {'reasoning_effort': 'low', 'top_p': 0.9} (values read as JSON if possible)."""
    if isinstance(items, dict):
        return dict(items)
    out = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise RuntimeError(f"--extra expects KEY=VALUE, got '{item}'")
        try:
            out[key.strip()] = json.loads(value)
        except ValueError:
            out[key.strip()] = value
    return out


def detect_provider(env) -> str | None:
    """Guess the provider from which API key variables are set (only if there is exactly one)."""
    found = [name for name, p in PROVIDERS.items() if any(env.get(v) for v in p["key_envs"])]
    if len(found) > 1:
        raise RuntimeError("Several API keys are set (" + ", ".join(found) + "). Choose one with --llm NAME.")
    if found:
        return found[0]
    return "custom" if env.get("LLM_BASE_URL") else None


def resolve_settings(provider: str | None, *, model=None, base_url=None, api_key=None, api_key_env=None,
                     api_format=None, extra=None, json_mode=False, settings_path=None, env=None) -> ResolvedLLM:
    """Combine flags, environment, settings file and provider defaults (see the module docstring)."""
    env = os.environ if env is None else env
    saved = load_settings_file(settings_path)
    name = (provider or "auto").strip().lower()
    if name == "auto":
        name = (env.get("LLM_PROVIDER") or saved.get("provider") or detect_provider(env) or "").lower()
        if not name:
            raise RuntimeError("No LLM provider is configured. Run scripts/setup_llm.py, or pass --llm NAME "
                               f"(one of: {', '.join(PROVIDERS)}), or set LLM_PROVIDER.")
    custom_url = base_url or env.get("LLM_BASE_URL") or (saved.get("base_url") if saved.get("provider") == name else None)
    preset = PROVIDERS.get(name)
    unlisted = preset is None
    if preset is None:                                    # unknown name: accept it if an address is known
        if not custom_url:
            raise RuntimeError(f"Unknown provider '{name}'. Use one of: {', '.join(PROVIDERS)}, or "
                               f"--llm custom --base-url URL for any OpenAI-compatible service.")
        preset = PROVIDERS["custom"]
    mine = saved if saved.get("provider") == name else {}          # saved values only for the same provider

    api = api_format or mine.get("api_format") or preset["api"]
    if api not in API_FORMATS + ("gemini",):
        raise RuntimeError(f"Unknown API format '{api}' (use openai or anthropic).")
    url = (custom_url or preset["base_url"] or "").rstrip("/") or None
    if api != "gemini" and not url:
        raise RuntimeError(f"Provider '{name}' needs an API address: pass --base-url URL or set LLM_BASE_URL.")
    mdl = model or env.get("LLM_MODEL") or mine.get("model")
    if api != "gemini" and not mdl:
        raise RuntimeError("No model name given. Pass --model NAME or set LLM_MODEL (see the provider's model list).")

    key, source = api_key, "--api-key"
    if not key:
        candidates = ([api_key_env] if api_key_env else []) + ([mine["api_key_env"]] if mine.get("api_key_env") else []) \
            + list(preset["key_envs"]) + ["LLM_API_KEY"]
        for var in candidates:
            if env.get(var):
                key, source = env[var], var
                break
        else:
            source = "none"
    if api_key_env and not env.get(api_key_env) and not api_key:
        raise RuntimeError(f"The environment variable {api_key_env} is not set.")
    if not key and name not in KEY_OPTIONAL and not unlisted:
        names = list(preset["key_envs"]) + ["LLM_API_KEY"]
        raise RuntimeError(f"No API key found for '{name}'. Set one of: {', '.join(names)} "
                           f"(Command Prompt: set {names[0]}=your_key), or pass --api-key-env NAME.")
    merged = {**parse_extra(mine.get("extra")), **parse_extra(extra)}
    return ResolvedLLM(name, api, url, mdl, key, source, merged, bool(json_mode))


# ------------------------------------------------------------------ HTTP client
class _APIError(Exception):
    def __init__(self, status: int, body: str, retry_after: str | None = None):
        super().__init__(f"HTTP {status}")
        self.status, self.body, self.retry_after = status, body, retry_after


class HTTPLLM:
    """Chat-completion client over plain HTTPS for the OpenAI-style and the Anthropic-style API.

    Requests are spaced out to stay under a per-minute quota (rpm), transient errors (rate limit, server
    busy, network) are retried with a pause, and settings that a model rejects (temperature, which token
    limit field to use, response_format) are dropped or swapped automatically.
    """

    def __init__(self, *, base_url: str, model: str, api: str = "openai", api_key: str | None = None,
                 provider: str = "custom", temperature: float | None = 0.0, max_tokens: int = 2048,
                 rpm: float | None = 10.0, max_retries: int = 5, timeout: float = 120.0, json_mode: bool = False,
                 extra: dict | None = None, sleep=time.sleep, clock=time.monotonic):
        if api not in API_FORMATS:
            raise ValueError("api must be 'openai' or 'anthropic'")
        self.api, self.provider, self.name = api, provider, model
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.temperature, self.max_tokens, self.json_mode = temperature, max_tokens, json_mode
        self.extra = dict(extra or {})
        self.interval = 60.0 / rpm if rpm else 0.0
        self.max_retries, self.timeout = max_retries, timeout
        self._sleep, self._clock, self._last_call = sleep, clock, None
        self.last_usage = None

    # -- request construction
    def _url(self) -> str:
        if self.api == "anthropic":
            return self.base_url + ("/messages" if self.base_url.endswith("/v1") else "/v1/messages")
        return self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": "llm-commander/1.0"}
        if self.api == "anthropic":
            h["anthropic-version"] = "2023-06-01"
            if self.api_key:
                h["x-api-key"] = self.api_key
        elif self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, system: str, user: str) -> dict:
        if self.api == "anthropic":
            body = {"model": self.name, "max_tokens": self.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
        else:
            body = {"model": self.name, "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra)
        return body

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(self._url(), data=json.dumps(payload).encode("utf-8"),
                                         headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise _APIError(e.code, e.read().decode("utf-8", "replace"), e.headers.get("Retry-After")) from None
        except (urllib.error.URLError, OSError, ValueError) as e:     # no connection, timeout, bad reply
            raise _APIError(0, str(e)) from None

    # -- response parsing
    def _read(self, data: dict) -> str:
        if self.api == "anthropic":
            usage = data.get("usage") or {}
            self.last_usage = {"in": usage.get("input_tokens", 0) or 0, "out": usage.get("output_tokens", 0) or 0}
            return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
        usage = data.get("usage") or {}
        self.last_usage = {"in": usage.get("prompt_tokens", 0) or 0, "out": usage.get("completion_tokens", 0) or 0}
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""
        if isinstance(content, list):                              # some services return a list of parts
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""

    # -- the call
    def _throttle(self) -> None:
        if self.interval and self._last_call is not None:
            wait = self.interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _clean(self, text: str) -> str:
        text = (text or "").strip()[:400]
        return text.replace(self.api_key, "***") if self.api_key else text

    def _explain(self, e: _APIError) -> str:
        where = f"{self.provider} ({self.base_url}), model '{self.name}'"
        if e.status == 0:
            return f"Could not reach {where}: {self._clean(e.body)}\nCheck the internet connection and the --base-url."
        hint = {401: "The API key was rejected: check the key and that it belongs to this provider.",
                403: "Access was refused: the key may lack permission for this model.",
                404: "Not found: check the model name and the --base-url.",
                429: "Rate limit or quota exceeded: wait, lower --rpm, or check the plan's quota."}.get(
                    e.status, "Check the key, the model name and the --base-url.")
        return f"{where} failed with HTTP {e.status}: {self._clean(e.body)}\n{hint}"

    def complete(self, system: str, user: str) -> str:
        payload = self._payload(system, user)
        network_failures = 0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                data = self._post(payload)
                break
            except _APIError as e:
                low = e.body.lower()
                if e.status in (400, 422):                       # the model rejects one of the settings: adjust
                    if "temperature" in low and "temperature" in payload:
                        payload.pop("temperature"); self.temperature = None
                        continue
                    if "max_completion_tokens" in low and "max_tokens" in payload:
                        payload["max_completion_tokens"] = payload.pop("max_tokens")
                        continue
                    if "max_completion_tokens" in low and "max_completion_tokens" in payload:
                        payload["max_tokens"] = payload.pop("max_completion_tokens")
                        continue
                    if "response_format" in low and "response_format" in payload:
                        payload.pop("response_format")
                        continue
                if e.status == 0:
                    network_failures += 1
                transient = e.status in (429, 500, 502, 503, 504) or (e.status == 0 and network_failures <= 2)
                if not transient or attempt == self.max_retries:
                    raise RuntimeError(self._explain(e)) from None
                try:
                    pause = float(e.retry_after) if e.retry_after else None
                except ValueError:
                    pause = None
                if pause is None:
                    pause = 2.0 * network_failures if e.status == 0 else 5.0 * 2 ** attempt
                self._sleep(min(90.0, pause))
        return self._read(data)


# ------------------------------------------------------------------ construction
def build_llm(provider: str | None, model: str | None = None, rpm: float | None = 10.0, *, base_url=None,
              api_key=None, api_key_env=None, api_format=None, extra=None, json_mode=False,
              settings_path=None, env=None):
    """LLM backend for a provider name ('auto' = from LLM_PROVIDER / settings file / the key that is set)."""
    r = resolve_settings(provider, model=model, base_url=base_url, api_key=api_key, api_key_env=api_key_env,
                         api_format=api_format, extra=extra, json_mode=json_mode, settings_path=settings_path, env=env)
    if r.api == "gemini":
        from .llm import GeminiLLM

        return GeminiLLM(model=r.model, rpm=rpm, api_key=r.api_key)
    return HTTPLLM(base_url=r.base_url, model=r.model, api=r.api, api_key=r.api_key, provider=r.provider,
                   rpm=rpm, json_mode=r.json_mode, extra=r.extra)


# ------------------------------------------------------------------ command-line helpers
def add_provider_arguments(parser) -> None:
    g = parser.add_argument_group("LLM provider (any service; see docs/llm_commander.md)")
    g.add_argument("--base-url", default=None, help="API address of an OpenAI-compatible service (env: LLM_BASE_URL)")
    g.add_argument("--api-key-env", default=None, help="name of the environment variable that holds the API key")
    g.add_argument("--api-key", default=None,
                   help="the API key itself (stays visible in the command history: an environment variable is safer)")
    g.add_argument("--api-format", choices=list(API_FORMATS), default=None,
                   help="request format for services not in the provider list (default: openai)")
    g.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="extra field sent with every request, e.g. reasoning_effort=low (repeatable)")
    g.add_argument("--json-mode", action="store_true", help="ask the service for JSON output (response_format)")


def provider_options(args) -> dict:
    return {"base_url": args.base_url, "api_key": args.api_key, "api_key_env": args.api_key_env,
            "api_format": args.api_format, "extra": args.extra, "json_mode": args.json_mode}
