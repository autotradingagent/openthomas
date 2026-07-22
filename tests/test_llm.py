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


# --- failover ------------------------------------------------------------------

def _primary_down_backup_up(primary_calls, backup_calls, primary_reply="0.5", backup_reply="0.6"):
    """Routes a MockTransport by host: primary refuses until `primary_up[0]`
    is flipped true; backup always answers."""
    primary_up = [False]

    def handler(request):
        if str(request.url).startswith("http://primary"):
            primary_calls.append(1)
            if not primary_up[0]:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"choices": [{"message": {"content": primary_reply}}]})
        backup_calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": backup_reply}}]})

    return handler, primary_up


def _failover_config(**kw):
    return ModelConfig(
        provider="openai", model="glm-5.2", base_url="http://primary/v1", retries=0,
        fallback_cooldown_s=300.0,
        fallback=ModelConfig(provider="openai", model="qwen3.6-27b", base_url="http://backup/v1"),
        **kw,
    )


def test_no_fallback_configured_still_raises_on_a_dead_endpoint(monkeypatch):
    """Baseline: without `fallback` set, behavior is exactly what it was before."""
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)

    def handler(request):
        raise httpx.ConnectError("down", request=request)

    client = CompletionClient(
        ModelConfig(provider="openai", model="glm-5.2", base_url="http://x/v1", retries=0),
        http=http_client(handler),
    )
    with pytest.raises(httpx.ConnectError):
        client.complete("s", "u")


def test_fails_over_to_backup_once_primary_is_exhausted(monkeypatch):
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    primary_calls, backup_calls = [], []
    handler, _ = _primary_down_backup_up(primary_calls, backup_calls)
    client = CompletionClient(_failover_config(), http=http_client(handler))

    assert client.complete("s", "u") == "0.6"
    assert len(primary_calls) == 1
    assert len(backup_calls) == 1
    assert client.status == {"active": "fallback", "model": "qwen3.6-27b"}


def test_stays_on_backup_within_cooldown_without_reprobing_primary(monkeypatch):
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    primary_calls, backup_calls = [], []
    handler, _ = _primary_down_backup_up(primary_calls, backup_calls)
    client = CompletionClient(_failover_config(), http=http_client(handler))

    client.complete("s", "u")  # fails over
    assert client.complete("s", "u") == "0.6"  # second call, still within cooldown
    assert len(primary_calls) == 1  # primary was not re-tried
    assert len(backup_calls) == 2


def test_recovers_to_primary_once_cooldown_elapses_and_primary_answers(monkeypatch):
    """The whole point: failover must not be sticky. Once the primary is back
    up and the cooldown has elapsed, the next call switches back on its own —
    nobody has to flip it back by hand."""
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    clock = {"t": 0.0}
    monkeypatch.setattr("openthomas.llm.time.monotonic", lambda: clock["t"])
    primary_calls, backup_calls = [], []
    handler, primary_up = _primary_down_backup_up(primary_calls, backup_calls)
    client = CompletionClient(_failover_config(), http=http_client(handler))

    assert client.complete("s", "u") == "0.6"  # primary down -> fails over
    assert client.status["active"] == "fallback"

    clock["t"] += 100  # within the 300s cooldown — stays on backup
    assert client.complete("s", "u") == "0.6"
    assert client.status["active"] == "fallback"

    primary_up[0] = True
    clock["t"] += 300  # cooldown elapsed — probes primary again
    assert client.complete("s", "u") == "0.5"
    assert client.status == {"active": "primary", "model": "glm-5.2"}


def test_status_sink_fires_only_on_transitions_not_every_call(monkeypatch):
    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    clock = {"t": 0.0}
    monkeypatch.setattr("openthomas.llm.time.monotonic", lambda: clock["t"])
    events = []
    handler, primary_up = _primary_down_backup_up([], [])
    client = CompletionClient(_failover_config(), http=http_client(handler), node="forecast",
                              status_sink=lambda **kw: events.append(kw))

    client.complete("s", "u")  # primary -> fallback: one event
    client.complete("s", "u")  # still degraded, no probe due yet: no new event
    assert [e["active"] for e in events] == ["fallback"]
    assert events[0]["node"] == "forecast"

    primary_up[0] = True
    clock["t"] += 300
    client.complete("s", "u")  # fallback -> primary: one more event
    assert [e["active"] for e in events] == ["fallback", "primary"]


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


def test_forecast_is_attributed_to_the_model_that_actually_answered(monkeypatch):
    """When the primary is down, the journal's `model` column must say so —
    not silently claim the primary produced a forecast the fallback made."""
    from openthomas.forecast.engine import ForecastEngine
    from openthomas.markets.base import Market

    monkeypatch.setattr("openthomas.llm.time.sleep", lambda *_: None)
    handler, _ = _primary_down_backup_up([], [], backup_reply='{"probability": 0.6}')
    cfg = _failover_config(ensemble_size=1)
    engine = ForecastEngine(cfg)
    engine.client = CompletionClient(cfg, http=http_client(handler))
    m = Market(id="x", platform="kalshi", question="?", yes_bid=0.4, yes_ask=0.42)

    forecast = engine.forecast(m)
    assert forecast.model == "qwen3.6-27b"  # not cfg.model ("glm-5.2")
