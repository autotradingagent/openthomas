"""Completion providers: one interface, four backends.

- "anthropic":  Anthropic Messages API
- "openai":     any OpenAI-compatible /chat/completions — OpenAI, OpenRouter,
                and local vLLM / Ollama / llama.cpp servers
- "claude-cli": Claude Code in print mode — billed to a Claude subscription
                instead of API credits
- "codex-cli":  OpenAI Codex in exec mode — billed to a ChatGPT subscription

Every LLM node (forecaster, reflector, …) takes its own ModelConfig, so
high-token work can run on a local server while the hardest judgments go to
a frontier API — or everything rides an existing subscription.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from .config import ModelConfig
from .memory.usage import Usage, now

# HTTP statuses worth retrying: the server is up but not ready (vLLM still
# loading the model → 503), momentarily overloaded (429), or a proxy blipped
# (502/504). A 400/401/404 is our fault and will not fix itself on retry.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def extract_json(text: str) -> dict | None:
    """First {...} blob in LLM output, parsed; None if absent or invalid.
    The one JSON-from-model-text extractor — forecaster, lesson curator, and
    evolution proposer must all parse replies the same way."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class CompletionError(RuntimeError):
    """A completion failed for infrastructure reasons (endpoint down, CLI
    missing, timeout). Callers treat it like a network error: skip the sample,
    never crash the cycle."""


class CompletionClient:
    def __init__(self, config: ModelConfig, http: httpx.Client | None = None,
                 run=subprocess.run, usage_sink=None, node: str = ""):
        """`usage_sink`: optional fn(Usage) -> None, called once per completion.
        `node` labels the caller in that ledger (forecast | reflect | …)."""
        self.config = config
        self.http = http or httpx.Client(timeout=config.timeout_s)
        self.run = run  # injectable for tests
        self.usage_sink = usage_sink
        self.node = node

    def _record(self, prompt_tokens=None, completion_tokens=None, cached_tokens=None) -> None:
        if self.usage_sink is None:
            return
        self.usage_sink(Usage(
            ts=now(), node=self.node, provider=self.config.provider,
            model=self.config.model, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cached_tokens=cached_tokens,
        ))

    def complete(self, system: str, user: str) -> str:
        c = self.config
        if c.provider == "anthropic":
            return self._anthropic(system, user)
        if c.provider == "openai":
            return self._openai(system, user)
        if c.provider == "claude-cli":
            return self._claude_cli(system, user)
        if c.provider == "codex-cli":
            return self._codex_cli(system, user)
        raise ValueError(
            f"unknown LLM provider {c.provider!r}; "
            "use anthropic | openai | claude-cli | codex-cli"
        )

    # --- HTTP providers ----------------------------------------------------------
    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        """POST with bounded retry, so a restarting endpoint is waited out, not
        dropped. Retries connection errors and RETRY_STATUS responses with
        exponential backoff; re-raises the last error once retries run out so
        the caller (forecaster, reflector) skips the sample as it always has.

        Local reasoning servers go away for tens of seconds when they reload a
        model or recover from an OOM. Without this, every forecast in that
        window returns None and the cycle trades nothing; with it, the sample
        blocks a few seconds and succeeds the moment the server is back."""
        c = self.config
        last: Exception | None = None
        for attempt in range(max(c.retries, 0) + 1):
            if attempt:
                # jitter breaks the lockstep of an N-sample ensemble all
                # retrying against the same recovering server at once.
                delay = min(c.retry_backoff_s * 2 ** (attempt - 1), c.retry_max_s)
                time.sleep(delay * (0.7 + 0.6 * random.random()))
            try:
                resp = self.http.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last = e
                if e.response.status_code not in RETRY_STATUS:
                    raise  # 4xx: our request is wrong; retrying cannot help
            except httpx.TransportError as e:
                last = e  # connect refused / timeout / reset — the server may be back soon
        raise last  # type: ignore[misc]

    def _anthropic(self, system: str, user: str) -> str:
        c = self.config
        body = self._post(
            (c.base_url or "https://api.anthropic.com") + "/v1/messages",
            {"x-api-key": c.api_key or "", "anthropic-version": "2023-06-01"},
            {
                "model": c.model, "max_tokens": c.max_tokens, "temperature": c.temperature,
                "system": system, "messages": [{"role": "user", "content": user}],
            },
        )
        u = body.get("usage") or {}
        self._record(u.get("input_tokens"), u.get("output_tokens"),
                     u.get("cache_read_input_tokens"))
        return body["content"][0]["text"]

    def _openai(self, system: str, user: str) -> str:
        body = self._post(
            (self.config.base_url or "https://api.openai.com/v1") + "/chat/completions",
            {"Authorization": f"Bearer {self.config.api_key or 'local'}"},
            {
                "model": self.config.model, "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **self.config.extra_body,
            },
        )
        u = body.get("usage") or {}
        self._record(u.get("prompt_tokens"), u.get("completion_tokens"),
                     (u.get("prompt_tokens_details") or {}).get("cached_tokens"))
        msg = body["choices"][0]["message"]
        # Reasoning models (GLM, DeepSeek) may return content=null when the
        # thinking budget runs out; surface whatever text exists, never None.
        return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

    # --- subscription CLI providers ------------------------------------------------
    def _run_cli(self, cmd: list[str], stdin: str) -> subprocess.CompletedProcess:
        try:
            proc = self.run(cmd, input=stdin, capture_output=True, text=True,
                            timeout=self.config.timeout_s)
        except FileNotFoundError:
            raise CompletionError(
                f"{cmd[0]!r} not found — install it and log in once, or switch provider"
            ) from None
        except subprocess.TimeoutExpired:
            raise CompletionError(f"{cmd[0]} timed out after {self.config.timeout_s}s") from None
        if proc.returncode != 0:
            raise CompletionError(f"{cmd[0]} exited {proc.returncode}: {proc.stderr[:300]}")
        self._record()  # subscription CLIs report no token counts
        return proc

    def _claude_cli(self, system: str, user: str) -> str:
        """`claude -p`: temperature/max_tokens don't apply; model may be an
        alias like 'sonnet' or empty for the CLI's default."""
        c = self.config
        cmd = [c.command or "claude", "-p", "--output-format", "text",
               "--system-prompt", system]
        if c.model:
            cmd += ["--model", c.model]
        return self._run_cli(cmd, user).stdout.strip()

    def _codex_cli(self, system: str, user: str) -> str:
        """`codex exec`: no system-prompt flag, so prepend it; the final
        answer is read from --output-last-message (stdout carries the
        session log)."""
        c = self.config
        with tempfile.NamedTemporaryFile(prefix="openthomas-codex-", suffix=".txt",
                                         delete=False) as f:
            out_path = Path(f.name)
        try:
            cmd = [c.command or "codex", "exec", "--skip-git-repo-check",
                   "-s", "read-only", "-o", str(out_path)]
            if c.model:
                cmd += ["-m", c.model]
            self._run_cli(cmd, f"{system}\n\n{user}" if system else user)
            answer = out_path.read_text().strip()
        finally:
            out_path.unlink(missing_ok=True)
        if not answer:
            raise CompletionError("codex exec produced no final message")
        return answer
