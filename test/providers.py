"""Resumable OpenRouter JSON calls with complete cache provenance."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _verified_ssl_context() -> ssl.SSLContext:
    """Use certifi on Python installations whose OpenSSL CA path is empty (common on macOS)."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


_SSL_CONTEXT = _verified_ssl_context()


class ProviderError(RuntimeError):
    pass


@dataclass
class ProviderResponse:
    data: dict[str, Any]
    usage: dict[str, Any]
    cache_hit: bool
    request_hash: str
    model: str
    provider: str
    actual_model: str | None = None
    route_provider: str | None = None


class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / max(float(requests_per_minute), 0.01)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            if delay:
                time.sleep(delay)
            self._next = time.monotonic() + self.interval


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ProviderError(f"Model geçerli JSON döndürmedi: {exc}") from exc
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner:
            raise ProviderError(f"Model JSON'u ayrıştırılamadı: {inner}") from inner
    if not isinstance(value, dict):
        raise ProviderError("Model çıktısının kökü JSON object olmalı")
    return value


class OpenRouterProvider:
    def __init__(self, spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
        self.spec = dict(spec)
        self.model = self.spec["model"]
        self.provider = "openrouter"
        self.cache_dir = cache_dir / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self.limiter = RateLimiter(self.spec.get("requests_per_minute", 20))
        self._write_lock = threading.Lock()

    def _identity(self, system: str, prompt: str, schema: dict, purpose: str) -> tuple[str, dict]:
        identity = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.spec.get("base_url"),
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "temperature": self.spec.get("temperature", 0.1),
            "max_tokens": self.spec.get("max_tokens", 5000),
            "provider_preferences": self.spec.get("provider_preferences", {}),
            "reasoning": self.spec.get("reasoning"),
            "plugins": self.spec.get("plugins", []),
            "pipeline": self.run_metadata,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity

    def call_json(self, system: str, prompt: str, schema: dict, purpose: str) -> ProviderResponse:
        request_hash, identity = self._identity(system, prompt, schema, purpose)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProviderResponse(
                cached["data"], cached.get("usage", {}), True, request_hash, self.model,
                self.provider, cached.get("response_model"), cached.get("route_provider"),
            )

        key = os.getenv(self.spec["api_key_env"])
        if not key:
            raise ProviderError(f"Eksik API anahtarı: {self.spec['api_key_env']}")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.spec.get("temperature", 0.1),
            "max_tokens": self.spec.get("max_tokens", 5000),
            "response_format": {"type": "json_schema", "json_schema": schema},
            "provider": {
                "require_parameters": True,
                **self.spec.get("provider_preferences", {}),
            },
        }
        if self.spec.get("reasoning"):
            body["reasoning"] = self.spec["reasoning"]
        if self.spec.get("plugins"):
            body["plugins"] = self.spec["plugins"]
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/TR-morph-retrieval/turkish-morph-retrieval",
            "X-Title": "Turkish Morph Retrieval Test Builder",
        }

        last_error: Exception | None = None
        for attempt in range(5):
            self.limiter.wait()
            req = urllib.request.Request(self.spec["base_url"], data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=float(self.spec.get("timeout_seconds", 180)),
                    context=_SSL_CONTEXT,
                ) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                choice = raw["choices"][0]
                content = choice["message"]["content"]
                try:
                    data = _extract_json(content)
                except ProviderError as exc:
                    raise ProviderError(
                        f"{exc}; model={raw.get('model', self.model)}, "
                        f"provider={raw.get('provider', 'unknown')}, "
                        f"finish_reason={choice.get('finish_reason', 'unknown')}, "
                        f"content_chars={len(str(content or ''))}"
                    ) from exc
                record = {
                    "identity": identity,
                    "data": data,
                    "usage": raw.get("usage", {}),
                    "response_model": raw.get("model", self.model),
                    "route_provider": raw.get("provider"),
                    "created_at_unix": int(time.time()),
                }
                tmp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
                with self._write_lock:
                    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp.replace(cache_path)
                return ProviderResponse(
                    data, record["usage"], False, request_hash, self.model, self.provider,
                    record["response_model"], record["route_provider"],
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = ProviderError(f"OpenRouter HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ProviderError) as exc:
                last_error = exc
            if attempt < 4:
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise ProviderError(f"OpenRouter çağrısı başarısız: {last_error}")


class CodexCliProvider:
    """Structured generation through the locally authenticated Codex CLI."""

    def __init__(self, spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
        self.spec = dict(spec)
        self.model = self.spec.get("model", "gpt-5.6-sol")
        self.provider = "codex_cli"
        self.cache_dir = cache_dir / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self._write_lock = threading.Lock()
        self.executable = self.spec.get("executable") or shutil.which("codex")
        if not self.executable:
            raise ProviderError("Codex CLI bulunamadı")

    def _identity(self, system: str, prompt: str, schema: dict, purpose: str) -> tuple[str, dict]:
        identity = {
            "provider": self.provider,
            "model": self.model,
            "authentication_mode": self.spec.get("authentication_mode", "saved_cli_login"),
            "reasoning_effort": self.spec.get("reasoning_effort", "medium"),
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "pipeline": self.run_metadata,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity

    def call_json(self, system: str, prompt: str, schema: dict, purpose: str) -> ProviderResponse:
        request_hash, identity = self._identity(system, prompt, schema, purpose)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProviderResponse(
                cached["data"], cached.get("usage", {}), True, request_hash,
                self.model, self.provider, cached.get("response_model", self.model),
                cached.get("route_provider", "codex_cli"),
            )
        if self.spec.get("cache_only"):
            raise ProviderError(f"Codex cache-only modunda yanıt bulunamadı: {request_hash}")

        request_dir = self.cache_dir / request_hash
        request_dir.mkdir(parents=True, exist_ok=True)
        schema_path = request_dir / "output_schema.json"
        output_path = request_dir / "last_message.json"
        schema_path.write_text(
            json.dumps(schema.get("schema", schema), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            str(self.executable), "exec",
            "--model", str(self.model),
            "--sandbox", "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--config", f'model_reasoning_effort="{self.spec.get("reasoning_effort", "medium")}"',
            "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
            "--cd", str(Path(self.spec.get("workdir", Path.cwd())).resolve()),
            "-",
        ]
        combined_prompt = (
            "SYSTEM INSTRUCTIONS\n" + system.strip() +
            "\n\nUSER TASK\n" + prompt.strip() +
            "\n\nReturn only the JSON value required by the output schema."
        )
        started = time.monotonic()
        try:
            process = subprocess.run(
                command,
                input=combined_prompt,
                text=True,
                capture_output=True,
                timeout=float(self.spec.get("timeout_seconds", 900)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Codex CLI zaman aşımı: {exc}") from exc
        if process.returncode != 0 or not output_path.exists():
            detail = (process.stderr or process.stdout)[-4000:]
            raise ProviderError(
                f"Codex CLI başarısız (exit={process.returncode}): {detail}"
            )
        data = _extract_json(output_path.read_text(encoding="utf-8"))
        usage = {
            "wall_seconds": round(time.monotonic() - started, 3),
            "cli_stdout_tail": process.stdout[-1000:],
        }
        record = {
            "identity": identity,
            "data": data,
            "usage": usage,
            "created_at_unix": int(time.time()),
        }
        tmp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        with self._write_lock:
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(cache_path)
        return ProviderResponse(
            data, usage, False, request_hash, self.model, self.provider,
            self.model, "codex_cli",
        )


class ClaudeCliProvider:
    """Structured generation through Claude Code's authenticated non-interactive mode."""

    def __init__(self, spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
        self.spec = dict(spec)
        self.model = self.spec["model"]
        self.provider = "claude_cli"
        self.cache_dir = cache_dir / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self._write_lock = threading.Lock()
        self.executable = self.spec.get("executable") or shutil.which("claude")
        if not self.executable:
            raise ProviderError(
                "Claude Code CLI bulunamadı; kurulumdan sonra `claude login` ile giriş yap"
            )

    def _identity(self, system: str, prompt: str, schema: dict, purpose: str) -> tuple[str, dict]:
        identity = {
            "provider": self.provider,
            "model": self.model,
            "authentication_mode": self.spec.get("authentication_mode", "saved_cli_login"),
            "reasoning_effort": self.spec.get("reasoning_effort", "medium"),
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "pipeline": self.run_metadata,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity

    def call_json(self, system: str, prompt: str, schema: dict, purpose: str) -> ProviderResponse:
        request_hash, identity = self._identity(system, prompt, schema, purpose)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProviderResponse(
                cached["data"], cached.get("usage", {}), True, request_hash,
                self.model, self.provider, cached.get("response_model", self.model),
                "claude_cli",
            )

        output_schema = schema.get("schema", schema)
        command = [
            str(self.executable), "-p",
            "--model", str(self.model),
            "--effort", str(self.spec.get("reasoning_effort", "medium")),
            "--output-format", "json",
            "--json-schema", json.dumps(output_schema, ensure_ascii=False, separators=(",", ":")),
            "--max-turns", "1",
            "--tools", "",
            "--no-session-persistence",
            "--permission-mode", "plan",
            "--system-prompt", system.strip(),
            prompt.strip() + "\n\nReturn only the JSON value required by the schema.",
        ]
        started = time.monotonic()
        try:
            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                cwd=str(Path(self.spec.get("workdir", Path.cwd())).resolve()),
                timeout=float(self.spec.get("timeout_seconds", 1800)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Claude Code CLI zaman aşımı: {exc}") from exc
        if process.returncode != 0:
            detail = (process.stderr or process.stdout)[-4000:]
            raise ProviderError(
                f"Claude Code CLI başarısız (exit={process.returncode}): {detail}"
            )
        try:
            envelope = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Claude Code CLI JSON envelope döndürmedi: {exc}") from exc
        if envelope.get("is_error") or envelope.get("subtype") not in {None, "success"}:
            raise ProviderError(
                f"Claude Code structured output başarısız: {envelope.get('subtype', 'unknown')}"
            )
        structured = envelope.get("structured_output")
        if isinstance(structured, dict):
            data = structured
        else:
            data = _extract_json(envelope.get("result", ""))

        model_usage = envelope.get("modelUsage") or envelope.get("model_usage") or {}
        actual_models = sorted(model_usage) if isinstance(model_usage, dict) else []
        if len(actual_models) > 1:
            raise ProviderError(
                "Claude Code birden fazla model kullandı; provenance için fallback kapatılmalı: "
                + ", ".join(actual_models)
            )
        actual_model = actual_models[0] if actual_models else self.model
        usage = {
            key: envelope[key]
            for key in ("total_cost_usd", "duration_ms", "duration_api_ms", "num_turns", "usage")
            if key in envelope
        }
        usage["wall_seconds"] = round(time.monotonic() - started, 3)
        if model_usage:
            usage["model_usage"] = model_usage
        record = {
            "identity": identity,
            "data": data,
            "usage": usage,
            "response_model": actual_model,
            "created_at_unix": int(time.time()),
        }
        tmp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        with self._write_lock:
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(cache_path)
        return ProviderResponse(
            data, usage, False, request_hash, self.model, self.provider,
            actual_model, "claude_cli",
        )


def make_provider(spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
    provider = spec.get("provider")
    if provider == "openrouter":
        return OpenRouterProvider(spec, cache_dir, run_metadata)
    if provider == "codex_cli":
        return CodexCliProvider(spec, cache_dir, run_metadata)
    if provider == "claude_cli":
        return ClaudeCliProvider(spec, cache_dir, run_metadata)
    raise ProviderError(f"Desteklenmeyen test provider'ı: {provider}")
