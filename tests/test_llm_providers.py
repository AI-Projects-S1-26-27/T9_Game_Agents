import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from strategies.llm_commander import HTTPLLM, build_llm, make_llm
from strategies.llm_commander.providers import (
    PROVIDERS, detect_provider, load_settings_file, parse_extra, resolve_settings,
)

ROOT = Path(__file__).resolve().parents[1]
ORDERS = json.dumps({"orders": [{"agent_id": "R3", "command": "scout", "target": "center", "rationale": "x"}]})


class FakeServer:
    """A tiny local stand-in for an LLM service. `script` holds (status, body, headers) answers used first;
    after that it answers like a real service in the OpenAI or the Anthropic format."""

    def __init__(self, reply_text: str = ORDERS):
        self.requests, self.script, self.reply_text = [], [], reply_text
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                if outer.script:
                    status, payload, headers = outer.script.pop(0)
                else:
                    status, headers = 200, {}
                    if self.path.endswith("/messages"):
                        payload = {"content": [{"type": "text", "text": outer.reply_text}],
                                   "usage": {"input_tokens": 5, "output_tokens": 2}}
                    else:
                        payload = {"choices": [{"message": {"content": outer.reply_text}}],
                                   "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
                data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def client(server, **kw):
    kw.setdefault("rpm", None)
    kw.setdefault("sleep", lambda s: None)
    return HTTPLLM(base_url=server.url + "/v1", model="m1", api_key="SECRET-KEY-123", **kw)


# ----------------------------------------------------------------------------- request / response formats
def test_openai_format_request_and_reply(server):
    llm = client(server)
    assert llm.complete("SYS", "USER") == ORDERS
    r = server.requests[0]
    assert r["path"] == "/v1/chat/completions"
    assert r["headers"]["authorization"] == "Bearer SECRET-KEY-123"
    assert r["body"]["model"] == "m1" and r["body"]["temperature"] == 0.0 and r["body"]["max_tokens"] == 2048
    assert r["body"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert "response_format" not in r["body"]
    assert llm.last_usage == {"in": 7, "out": 3}


def test_anthropic_format_request_and_reply(server):
    llm = client(server, api="anthropic")
    assert llm.complete("SYS", "USER") == ORDERS
    r = server.requests[0]
    assert r["path"] == "/v1/messages"
    assert r["headers"]["x-api-key"] == "SECRET-KEY-123" and "anthropic-version" in r["headers"]
    assert r["body"]["system"] == "SYS" and r["body"]["messages"] == [{"role": "user", "content": "USER"}]
    assert r["body"]["max_tokens"] == 2048
    assert llm.last_usage == {"in": 5, "out": 2}


def test_reply_given_as_list_of_parts_and_empty_reply(server):
    server.script = [(200, {"choices": [{"message": {"content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cd"}]}}]}, {}),
                     (200, {"choices": []}, {})]
    llm = client(server)
    assert llm.complete("s", "u") == "abcd"
    assert llm.complete("s", "u") == ""            # unreadable reply: the commander treats it as a failed reply


def test_extra_fields_and_json_mode_are_sent(server):
    llm = client(server, extra=parse_extra(["reasoning_effort=low", "top_p=0.5"]), json_mode=True)
    llm.complete("s", "u")
    body = server.requests[0]["body"]
    assert body["reasoning_effort"] == "low" and body["top_p"] == 0.5 and body["response_format"] == {"type": "json_object"}


# ----------------------------------------------------------------------------- errors and adjustments
def test_rejected_settings_are_adjusted_automatically(server):
    server.script = [
        (400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}}, {}),
        (400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.0 with this model."}}, {}),
        (400, {"error": {"message": "this model does not support response_format"}}, {}),
    ]
    llm = client(server, json_mode=True)
    assert llm.complete("s", "u") == ORDERS
    final = server.requests[-1]["body"]
    assert "max_tokens" not in final and final["max_completion_tokens"] == 2048
    assert "temperature" not in final and "response_format" not in final


def test_rate_limit_is_retried_using_retry_after(server):
    pauses = []
    server.script = [(429, {"error": "slow down"}, {"Retry-After": "2"})]
    llm = client(server, sleep=pauses.append)
    assert llm.complete("s", "u") == ORDERS
    assert pauses == [2.0] and len(server.requests) == 2


def test_wrong_key_gives_a_clear_message_and_never_prints_the_key(server):
    server.script = [(401, {"error": {"message": "Incorrect API key provided: SECRET-KEY-123"}}, {})]
    with pytest.raises(RuntimeError) as err:
        client(server).complete("s", "u")
    assert "key was rejected" in str(err.value) and "SECRET-KEY-123" not in str(err.value)


def test_unknown_model_message(server):
    server.script = [(404, {"error": "model not found"}, {})]
    with pytest.raises(RuntimeError, match="model name"):
        client(server).complete("s", "u")


def test_unreachable_address_fails_clearly_after_few_retries():
    pauses = []
    llm = HTTPLLM(base_url="http://127.0.0.1:9/v1", model="m", rpm=None, sleep=pauses.append, timeout=2)
    with pytest.raises(RuntimeError, match="Could not reach"):
        llm.complete("s", "u")
    assert len(pauses) <= 2


def test_requests_are_spaced_by_rpm(server):
    now, pauses = [0.0], []
    llm = HTTPLLM(base_url=server.url + "/v1", model="m", rpm=6, sleep=pauses.append, clock=lambda: now[0])
    llm.complete("s", "u")
    now[0] = 3.0
    llm.complete("s", "u")
    assert pauses == [7.0]


# ----------------------------------------------------------------------------- settings resolution
def test_precedence_flags_then_environment_then_file_then_preset(tmp_path):
    saved = tmp_path / "s.json"
    saved.write_text(json.dumps({"provider": "openai", "model": "file-model", "base_url": "http://file/v1"}))
    env = {"OPENAI_API_KEY": "k"}
    r = resolve_settings("openai", settings_path=saved, env=env)
    assert r.model == "file-model" and r.base_url == "http://file/v1"
    r = resolve_settings("openai", settings_path=saved, env={**env, "LLM_MODEL": "env-model", "LLM_BASE_URL": "http://env/v1"})
    assert r.model == "env-model" and r.base_url == "http://env/v1"
    r = resolve_settings("openai", model="flag-model", base_url="http://flag/v1/", settings_path=saved, env={**env, "LLM_MODEL": "env-model"})
    assert r.model == "flag-model" and r.base_url == "http://flag/v1"
    r = resolve_settings("openai", model="m", settings_path=tmp_path / "none.json", env=env)
    assert r.base_url == PROVIDERS["openai"]["base_url"]


def test_saved_values_never_leak_into_another_provider(tmp_path):
    saved = tmp_path / "s.json"
    saved.write_text(json.dumps({"provider": "openai", "model": "gpt-x", "base_url": "http://file/v1"}))
    r = resolve_settings("groq", model="llama", settings_path=saved, env={"GROQ_API_KEY": "k"})
    assert r.model == "llama" and r.base_url == PROVIDERS["groq"]["base_url"]


def test_key_sources_in_order_and_clear_errors(tmp_path):
    none = tmp_path / "none.json"
    assert resolve_settings("openai", model="m", api_key="flag", settings_path=none, env={"OPENAI_API_KEY": "a"}).api_key == "flag"
    assert resolve_settings("openai", model="m", api_key_env="MY_KEY", settings_path=none,
                            env={"MY_KEY": "mine", "OPENAI_API_KEY": "a"}).api_key == "mine"
    assert resolve_settings("openai", model="m", settings_path=none, env={"OPENAI_API_KEY": "a", "LLM_API_KEY": "g"}).api_key == "a"
    assert resolve_settings("openai", model="m", settings_path=none, env={"LLM_API_KEY": "g"}).api_key == "g"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_settings("openai", model="m", settings_path=none, env={})
    with pytest.raises(RuntimeError, match="MY_KEY"):
        resolve_settings("openai", model="m", api_key_env="MY_KEY", settings_path=none, env={})


def test_keyless_local_providers_and_required_fields(tmp_path):
    none = tmp_path / "none.json"
    r = resolve_settings("ollama", model="llama3", settings_path=none, env={})
    assert r.api_key is None and r.base_url.startswith("http://localhost")
    with pytest.raises(RuntimeError, match="model"):
        resolve_settings("ollama", settings_path=none, env={})
    with pytest.raises(RuntimeError, match="--base-url"):
        resolve_settings("custom", model="m", settings_path=none, env={})
    with pytest.raises(RuntimeError, match="Unknown provider"):
        resolve_settings("nonsense", model="m", settings_path=none, env={})
    assert resolve_settings("my-host", model="m", base_url="http://x/v1", settings_path=none, env={}).provider == "my-host"


def test_auto_detection(tmp_path):
    none = tmp_path / "none.json"
    assert detect_provider({"GROQ_API_KEY": "k"}) == "groq"
    assert detect_provider({"LLM_BASE_URL": "http://x"}) == "custom"
    assert detect_provider({}) is None
    with pytest.raises(RuntimeError, match="Several API keys"):
        detect_provider({"GROQ_API_KEY": "k", "OPENAI_API_KEY": "k"})
    assert resolve_settings("auto", model="m", settings_path=none, env={"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k"}).provider == "openai"
    with pytest.raises(RuntimeError, match="setup_llm"):
        resolve_settings("auto", settings_path=none, env={})


def test_gemini_settings_and_describe_masks_the_key(tmp_path):
    r = resolve_settings("gemini", settings_path=tmp_path / "none.json", env={"GOOGLE_API_KEY": "abcdef123456"})
    assert r.api == "gemini" and r.model is None and r.api_key == "abcdef123456"
    assert "abcdef123456" not in r.describe() and "3456" in r.describe()


def test_parse_extra_reads_json_values():
    assert parse_extra(["a=1", "b=x", "c=true", "d=[1,2]"]) == {"a": 1, "b": "x", "c": True, "d": [1, 2]}
    with pytest.raises(RuntimeError):
        parse_extra(["novalue"])


def test_build_llm_and_make_llm(server, tmp_path):
    llm = build_llm("custom", model="m", rpm=None, base_url=server.url + "/v1", api_key="k", settings_path=tmp_path / "x.json")
    assert isinstance(llm, HTTPLLM) and llm.provider == "custom" and llm.name == "m"
    cached = make_llm("custom", "m", None, str(tmp_path / "cache.json"), base_url=server.url + "/v1", api_key="k")
    assert cached.complete("s", "u") == ORDERS and cached.complete("s", "u") == ORDERS
    assert len(server.requests) == 1 and cached.hits == 1          # the second answer came from the cache
    assert make_llm("mock").name == "mock"


# ----------------------------------------------------------------------------- whole scripts, any provider
def run(script, *args, env=None):
    full = {**os.environ, **{k: v for k, v in (env or {}).items()}}
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER",
                "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_SETTINGS_FILE"):
        if var not in (env or {}):
            full.pop(var, None)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], capture_output=True, text=True,
                          cwd=ROOT, env=full)


def test_game_runs_with_flags_for_a_custom_service(server):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    r = run("run_llm_commander.py", "--llm", "custom", "--base-url", server.url + "/v1", "--model", "m",
            "--api-key", "SECRET-KEY-123", "--rpm", "6000", "--quiet", "--opponent", "balanced", "--log", "logs/_t.jsonl")
    assert r.returncode == 0, r.stderr
    assert "Result:" in r.stdout and "SECRET-KEY-123" not in r.stdout + r.stderr
    assert server.requests and all(q["headers"]["authorization"] == "Bearer SECRET-KEY-123" for q in server.requests)
    for f in (ROOT / "logs").glob("_t_game*.jsonl"):
        f.unlink()


def test_game_runs_with_environment_variables_only(server):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    r = run("run_llm_commander.py", "--llm", "auto", "--rpm", "6000", "--quiet", "--opponent", "balanced", "--log", "",
            env={"LLM_PROVIDER": "custom", "LLM_BASE_URL": server.url + "/v1", "LLM_MODEL": "m", "LLM_API_KEY": "ENV-KEY"})
    assert r.returncode == 0, r.stderr
    assert server.requests and server.requests[0]["headers"]["authorization"] == "Bearer ENV-KEY"


def test_anthropic_format_service_through_the_scripts(server):
    r = run("check_llm.py", "--llm", "custom", "--api-format", "anthropic", "--base-url", server.url, "--model", "claude-like",
            "--api-key", "K", "--rpm", "6000")
    assert r.returncode == 0, r.stderr + r.stdout
    assert any(q["path"] == "/v1/messages" and q["headers"].get("x-api-key") == "K" for q in server.requests)


def test_experiment_runner_accepts_any_provider(server, tmp_path):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    r = run("run_experiments.py", "--llm", "custom", "--base-url", server.url + "/v1", "--model", "m", "--api-key", "K",
            "--rpm", "6000", "--games", "2", "--conditions", "llm", "--opponents", "balanced", "--out", str(tmp_path / "g.csv"),
            "--summary", str(tmp_path / "s.csv"), "--log-dir", str(tmp_path / "logs"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert (tmp_path / "g.csv").read_text().count("\n") == 3 and server.requests


def test_setup_script_saves_settings_without_the_key_and_auto_uses_them(server, tmp_path):
    settings = tmp_path / "llm.json"
    r = run("setup_llm.py", "--provider", "custom", "--base-url", server.url + "/v1", "--model", "m", "--rpm", "6000",
            "--settings-path", str(settings), env={"LLM_API_KEY": "SECRET-KEY-123"})
    assert r.returncode == 0, r.stderr + r.stdout
    saved = json.loads(settings.read_text())
    assert saved == {"provider": "custom", "model": "m", "base_url": server.url + "/v1"}
    assert "SECRET-KEY-123" not in settings.read_text() + r.stdout and "set LLM_API_KEY=your_key" in r.stdout
    c = run("check_llm.py", "--llm", "auto", "--rpm", "6000", env={"LLM_SETTINGS_FILE": str(settings), "LLM_API_KEY": "SECRET-KEY-123"})
    assert c.returncode == 0, c.stderr + c.stdout
    assert load_settings_file(settings)["model"] == "m"


def test_setup_script_reports_a_failed_connection_and_saves_nothing(tmp_path):
    settings = tmp_path / "llm.json"
    r = run("setup_llm.py", "--provider", "custom", "--base-url", "http://127.0.0.1:9/v1", "--model", "m",
            "--settings-path", str(settings))
    assert r.returncode != 0 and "FAILED" in r.stderr and not settings.exists()


def test_clear_error_when_nothing_is_configured():
    r = run("check_llm.py")
    assert r.returncode != 0 and "setup_llm.py" in r.stderr
