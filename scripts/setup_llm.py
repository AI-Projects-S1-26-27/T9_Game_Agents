#!/usr/bin/env python3
"""Choose and test the LLM provider once; the other scripts then work with --llm auto.

  uv run python scripts/setup_llm.py                     # interactive (asks for provider, model, key)
  uv run python scripts/setup_llm.py --show              # what is configured now
  uv run python scripts/setup_llm.py --provider custom --base-url http://localhost:8000/v1 --model NAME --no-save

The API key is only used in memory for the connection test. It is never written to a file: the script
tells you which environment variable to set (Command Prompt: set NAME=your_key). Only provider, model and
address are saved, in .llm_settings.json (add it to .gitignore).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategies.llm_commander.providers import (  # noqa: E402
    KEY_OPTIONAL, PROVIDERS, SETTINGS_FILE, add_provider_arguments, build_llm, load_settings_file, parse_extra,
    resolve_settings,
)


def ask(prompt: str, default: str | None = None) -> str:
    text = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return text or (default or "")


def choose_provider() -> str:
    names = list(PROVIDERS)
    print("Providers:")
    for i, name in enumerate(names, 1):
        note = {"gemini": "Google Gemini", "custom": "any other service with an OpenAI-compatible API (you give the address)",
                "ollama": "local models, no key", "lmstudio": "local models, no key"}.get(name, "")
        print(f"  {i:>2}) {name:<11}{note}")
    while True:
        answer = ask("Number or name").lower()
        if answer in PROVIDERS:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("  Please choose one of the listed providers.")


def show() -> None:
    saved = load_settings_file(args.settings_path)
    print(f"Settings file ({args.settings_path or os.environ.get('LLM_SETTINGS_FILE') or SETTINGS_FILE}): "
          f"{json.dumps(saved) if saved else 'none'}")
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        print(f"  {var} = {os.environ.get(var, '(not set)')}")
    try:
        print("Resolved now:", resolve_settings("auto", settings_path=args.settings_path).describe())
    except RuntimeError as e:
        print("Resolved now:", e)


def main() -> None:
    global args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default=None, help=f"one of: {', '.join(PROVIDERS)}")
    p.add_argument("--model", default=None)
    p.add_argument("--rpm", type=float, default=10.0)
    p.add_argument("--settings-path", default=None, help=f"where to save (default: {SETTINGS_FILE})")
    p.add_argument("--no-test", action="store_true", help="skip the connection test")
    p.add_argument("--no-save", action="store_true", help="do not write the settings file")
    p.add_argument("--show", action="store_true", help="show the current configuration and exit")
    add_provider_arguments(p)
    args = p.parse_args()
    if args.show:
        return show()

    interactive = args.provider is None
    provider = args.provider or choose_provider()
    if provider not in PROVIDERS:
        sys.exit(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    preset = PROVIDERS[provider]
    model = args.model or (ask("Model name (as listed by the provider)") if interactive else None)
    base_url = args.base_url
    if interactive and (provider == "custom" or not preset["base_url"] and preset["api"] != "gemini"):
        base_url = ask("API address (for example https://host/v1)")
    api_format = args.api_format
    if interactive and provider == "custom":
        api_format = ask("Request format (openai or anthropic)", "openai")

    key = args.api_key
    env_name = args.api_key_env or (preset["key_envs"][0] if preset["key_envs"] else "LLM_API_KEY")
    options = dict(base_url=base_url, api_key_env=args.api_key_env, api_format=api_format, extra=args.extra,
                   json_mode=args.json_mode)
    if not key and not any(os.environ.get(v) for v in ([args.api_key_env] if args.api_key_env else []) + list(preset["key_envs"]) + ["LLM_API_KEY"]):
        if provider not in KEY_OPTIONAL and interactive:
            print(f"\nNo API key found in the environment. Paste it now to test it (typing is hidden; it is not saved).")
            key = getpass.getpass("API key: ").strip() or None
        elif provider not in KEY_OPTIONAL:
            print(f"Note: no API key found. Set {env_name} first (Command Prompt: set {env_name}=your_key).")

    try:
        resolved = resolve_settings(provider, model=model, api_key=key, settings_path=args.settings_path, **options)
    except RuntimeError as e:
        sys.exit(f"\n{e}")
    print("\nSettings:", resolved.describe())

    if not args.no_test:
        print("Testing the connection ...")
        try:
            llm = build_llm(provider, model=model, rpm=args.rpm, api_key=key, settings_path=args.settings_path, **options)
            t0 = time.time()
            reply = llm.complete("Reply with JSON only.", 'Return exactly {"ok": true}')
        except RuntimeError as e:
            sys.exit(f"FAILED: {e}\nNothing was saved. Fix the problem and run this script again.")
        except ImportError as e:
            sys.exit(f"FAILED: {e}\nGemini needs the google-genai package: run this with  uv run --with google-genai python scripts/setup_llm.py")
        print(f"OK: reply {reply.strip()[:80]!r} in {time.time() - t0:.1f}s, tokens {getattr(llm, 'last_usage', None)}")

    if not args.no_save:
        saved = {"provider": provider}
        if model:
            saved["model"] = model
        if base_url:
            saved["base_url"] = base_url
        if api_format:
            saved["api_format"] = api_format
        if args.api_key_env:
            saved["api_key_env"] = args.api_key_env
        if args.extra:
            saved["extra"] = parse_extra(args.extra)
        path = Path(args.settings_path or os.environ.get("LLM_SETTINGS_FILE") or SETTINGS_FILE)
        path.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"\nSaved provider/model/address to {path} (no key inside). Add .llm_settings.json to .gitignore.")

    if provider not in KEY_OPTIONAL or key or os.environ.get(env_name):
        print(f"\nTo use it, set the key in each Command Prompt window (or once for good with setx):\n"
              f"  set {env_name}=your_key\n  setx {env_name} your_key        (permanent; applies to NEW windows)\n"
              f"PowerShell:  $env:{env_name}=\"your_key\"      Mac/Linux:  export {env_name}=your_key")
    print("\nThen run, for example:\n  uv run python scripts/check_llm.py\n"
          "  uv run python scripts/run_llm_commander.py --llm auto --opponent classical --cache logs/cache.json")


if __name__ == "__main__":
    main()
