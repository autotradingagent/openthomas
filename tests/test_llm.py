import subprocess
from pathlib import Path

import httpx
import pytest

from openthomas.config import ModelConfig
from openthomas.llm import CompletionClient, CompletionError


def http_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- HTTP providers ---------------------------------------------------------

def test_openai_provider_hits_chat_completions():
    def handler(request):
        assert request.url == "http://localhost:8000/v1/chat/completions"
        body = request.read().decode()
        assert "glm-5.2" in body and "system" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "p=0.4"}}]})

    client = CompletionClient(
        ModelConfig(provider="openai", model="glm-5.2", base_url="http://localhost:8000/v1"),
        http=http_client(handler),
    )
    assert client.complete("sys", "user") == "p=0.4"


def test_openai_retries_through_a_restarting_endpoint(monkeypatch):
    """vLLM refusing connections then answering 503 while it reloads must be
    ridden through, not dropped — the sample should still return."""
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)  # no real backoff
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        if calls["n"] == 2:
            return httpx.Response(503, text="model loading")
        return httpx.Response(200, json={"choices": [{"message": {"content": "0.4"}}]})

    client = CompletionClient(
        ModelConfig(provider="openai", model="glm-5.2", base_url="http://x/v1", retries=4),
        http=http_client(handler),
    )
    assert client.complete("s", "u") == "0.4"
    assert calls["n"] == 3  # connect-error, 503, then success


def test_openai_gives_up_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("down", request=request)

    client = CompletionClient(
        ModelConfig(provider="openai", model="m", base_url="http://x/v1", retries=2),
        http=http_client(handler),
    )
    with pytest.raises(httpx.ConnectError):
        client.complete("s", "u")
    assert calls["n"] == 3  # initial try + 2 retries


def test_openai_does_not_retry_a_client_error(monkeypatch):
    """A 400 is our mistake; retrying wastes the recovering window on a request
    that will fail the same way every time."""
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = CompletionClient(
        ModelConfig(provider="openai", model="m", base_url="http://x/v1", retries=4),
        http=http_client(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("s", "u")
    assert calls["n"] == 1  # no retry on 4xx


def test_anthropic_provider_hits_messages():
    def handler(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["anthropic-version"]
        return httpx.Response(200, json={"content": [{"text": "hello"}]})

    client = CompletionClient(ModelConfig(provider="anthropic"), http=http_client(handler))
    assert client.complete("sys", "user") == "hello"


def test_unknown_provider_raises():
    client = CompletionClient(ModelConfig(provider="gemini"))
    with pytest.raises(ValueError, match="unknown LLM provider"):
        client.complete("s", "u")


# --- CLI providers ------------------------------------------------------------

class StubRun:
    """Captures the subprocess call; scriptable result."""

    def __init__(self, stdout="", returncode=0, raises=None, write_last_message=None):
        self.stdout, self.returncode = stdout, returncode
        self.raises, self.write_last_message = raises, write_last_message
        self.cmd, self.stdin = None, None

    def __call__(self, cmd, input=None, timeout=None, **kw):
        self.cmd, self.stdin = cmd, input
        if self.raises:
            raise self.raises
        if self.write_last_message is not None and "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text(self.write_last_message)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, "boom")


def test_claude_cli_flags_and_stdin():
    run = StubRun(stdout="0.42\n")
    client = CompletionClient(ModelConfig(provider="claude-cli", model="sonnet"), run=run)
    assert client.complete("be terse", "what is p?") == "0.42"
    assert run.cmd[:4] == ["claude", "-p", "--output-format", "text"]
    assert run.cmd[run.cmd.index("--system-prompt") + 1] == "be terse"
    assert run.cmd[run.cmd.index("--model") + 1] == "sonnet"
    assert run.stdin == "what is p?"


def test_claude_cli_empty_model_omits_flag():
    run = StubRun(stdout="ok")
    CompletionClient(ModelConfig(provider="claude-cli", model=""), run=run).complete("s", "u")
    assert "--model" not in run.cmd


def test_codex_cli_reads_last_message_file():
    run = StubRun(write_last_message="the answer")
    client = CompletionClient(ModelConfig(provider="codex-cli", model="gpt-5.2"), run=run)
    assert client.complete("sys", "usr") == "the answer"
    assert run.cmd[:3] == ["codex", "exec", "--skip-git-repo-check"]
    assert run.stdin == "sys\n\nusr"  # no system-prompt flag: prepended
    # temp output file is cleaned up
    assert not Path(run.cmd[run.cmd.index("-o") + 1]).exists()


def test_cli_missing_binary_is_completion_error():
    run = StubRun(raises=FileNotFoundError())
    client = CompletionClient(ModelConfig(provider="claude-cli"), run=run)
    with pytest.raises(CompletionError, match="not found"):
        client.complete("s", "u")


def test_cli_timeout_is_completion_error():
    run = StubRun(raises=subprocess.TimeoutExpired("claude", 300))
    client = CompletionClient(ModelConfig(provider="claude-cli"), run=run)
    with pytest.raises(CompletionError, match="timed out"):
        client.complete("s", "u")


def test_cli_nonzero_exit_is_completion_error():
    run = StubRun(returncode=1)
    client = CompletionClient(ModelConfig(provider="claude-cli"), run=run)
    with pytest.raises(CompletionError, match="exited 1"):
        client.complete("s", "u")


# --- engine integration --------------------------------------------------------

def test_forecast_survives_cli_failure():
    """A dead CLI provider must skip the sample, not crash the cycle."""
    from openthomas.forecast.engine import ForecastEngine
    from openthomas.markets.base import Market

    cfg = ModelConfig(provider="claude-cli", ensemble_size=2)
    engine = ForecastEngine(cfg)
    engine.client = CompletionClient(cfg, run=StubRun(raises=FileNotFoundError()))
    m = Market(id="x", platform="kalshi", question="?", yes_bid=0.4, yes_ask=0.42)
    assert engine.forecast(m) is None
