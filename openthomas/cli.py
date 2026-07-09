"""OpenThomas CLI: init, scan, run, report, vital."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import ModelConfig, RiskProfile, Settings

app = typer.Typer(
    name="openthomas",
    help="Autonomous AI trading agent for prediction markets (Polymarket, Kalshi).",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    bankroll: float = typer.Option(1000.0, help="USD the agent may deploy"),
    risk: str = typer.Option("conservative", help="conservative | moderate | aggressive"),
    goal: str = typer.Option("Grow the bankroll steadily; protecting capital beats chasing returns."),
    provider: str = typer.Option("anthropic", help="anthropic | openai (incl. local endpoints)"),
    model: str = typer.Option("claude-sonnet-5", help="forecasting model id"),
    base_url: str = typer.Option(None, help="custom endpoint, e.g. http://localhost:11434/v1"),
):
    """Create ~/.openthomas/config.yaml with your bankroll, goal, and risk profile."""
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    settings = Settings(
        bankroll=bankroll, goal=goal, risk=RiskProfile.preset(risk),
        forecaster=ModelConfig(provider=provider, model=model, base_url=base_url,
                               api_key_env=key_env),
    )
    path = settings.save()
    console.print(f"[green]✓[/green] Config written to {path}")
    console.print(f"  bankroll ${bankroll:,.0f} · risk={risk} · model={model}")
    console.print("  Mode is [bold]paper[/bold] (simulated fills on real prices). "
                  "Run [bold]openthomas run[/bold] to start.")


@app.command()
def scan(limit: int = typer.Option(20, help="max rows to show")):
    """Scan live markets and show tradeable candidates + cross-platform arbs."""
    from .agent.loop import build_connectors
    from .edge.scanner import EdgeScanner

    s = Settings.load()
    markets = []
    for connector in build_connectors(s.platforms).values():
        with console.status(f"fetching {connector.platform} markets…"):
            try:
                markets += connector.list_markets(limit=150)
            except Exception as e:
                console.print(f"[red]{connector.platform}: {e}[/red]")
    result = EdgeScanner(s.risk).scan(markets)

    table = Table(title=f"Candidates ({len(result.candidates)} of {len(markets)} markets pass filters)")
    for col in ("platform", "question", "bid", "ask", "vol 24h", "closes in"):
        table.add_column(col)
    for m in result.candidates[:limit]:
        hours = m.hours_to_close()
        table.add_row(
            m.platform, m.question[:70],
            f"{m.yes_bid:.2f}" if m.yes_bid is not None else "—",
            f"{m.yes_ask:.2f}" if m.yes_ask is not None else "—",
            f"${m.volume_24h:,.0f}",
            f"{hours:.0f}h" if hours is not None else "—",
        )
    console.print(table)
    console.print(f"Skipped: {result.skipped}")
    if result.arbs:
        console.print("\n[bold]Cross-platform arbitrage candidates[/bold] (verify resolution rules!)")
        for arb in result.arbs[:10]:
            console.print(f"  {arb.describe()}")


def _print_report(report) -> None:
    console.rule(f"cycle · account ${report.account_value:,.2f} · cash ${report.cash:,.2f}")
    console.print(f"markets {report.markets_seen} → candidates {report.candidates} "
                  f"→ forecasts {report.forecasts} → trades {len(report.trades)}")
    for t in report.trades:
        console.print(f"  [green]TRADE[/green] {t}")
    for s_ in report.settlements:
        console.print(f"  [cyan]SETTLED[/cyan] {s_}")
    for a in report.arbs:
        console.print(f"  [magenta]ARB?[/magenta] {a}")
    for r in report.rejections[:8]:
        console.print(f"  [dim]skip: {r}[/dim]")
    if report.halted:
        console.print("[red bold]KILL-SWITCH: max drawdown reached. Trading halted — "
                      "review the journal, then delete peak_value to resume.[/red bold]")


@app.command()
def run(
    once: bool = typer.Option(False, "--once", help="run a single cycle and exit"),
    live: bool = typer.Option(False, "--live", help="trade with real money (default: paper)"),
):
    """Run the trading loop (paper mode by default)."""
    from .agent.loop import Agent

    s = Settings.load()
    if live:
        if s.mode != "live":
            console.print("[red]Refusing --live: set `mode: live` in ~/.openthomas/config.yaml "
                          "as well, so live trading requires two explicit steps.[/red]")
            raise typer.Exit(1)
        console.print("[yellow bold]LIVE MODE — real money.[/yellow bold]")
    else:
        s.mode = "paper"
    agent = Agent(s)
    if once:
        _print_report(agent.cycle())
    else:
        console.print(f"Trading loop started · every {s.cycle_minutes}m · Ctrl-C to stop")
        final = agent.run_forever(on_report=_print_report)
        if final.halted:
            raise typer.Exit(3)  # distinct code so supervisors don't blind-restart


@app.command()
def report():
    """Performance summary: PnL, win rate, calibration, per-category stats."""
    from .forecast.calibration import brier_score, calibration_table
    from .memory.journal import Journal

    s = Settings.load()
    j = Journal(s.db_path)
    stats = j.settlement_stats()
    curve = j.equity_curve()
    value = curve[-1][1] if curve else s.bankroll
    console.print(f"[bold]Account value:[/bold] ${value:,.2f}  "
                  f"(start ${s.bankroll:,.2f}, {(value / s.bankroll - 1):+.1%})")
    console.print(f"Settled: {stats['n']} · win rate {stats['win_rate']:.0%} · "
                  f"avg win ${stats['avg_win']:.2f} / avg loss ${stats['avg_loss']:.2f}")
    pairs = j.forecast_outcome_pairs()
    if pairs:
        console.print(f"Brier score: {brier_score(pairs):.3f} (0.25 = coin flip, lower is better)")
        table = Table(title="Calibration")
        for col in ("forecast", "n", "observed"):
            table.add_column(col)
        for row in calibration_table(pairs):
            if row["n"]:
                table.add_row(row["bucket"], str(row["n"]),
                              f"{row['observed']:.0%}" if row["observed"] is not None else "—")
        console.print(table)
    cats = j.category_stats()
    if cats:
        table = Table(title="By category")
        for col in ("category", "settled", "win rate", "pnl"):
            table.add_column(col)
        for c in cats:
            table.add_row(c["category"] or "—", str(c["n"]),
                          f"{c['win_rate']:.0%}", f"${c['pnl']:+.2f}")
        console.print(table)

    from .report.brier import summarize_skill, weather_skill
    buckets = weather_skill(j)
    if buckets:
        table = Table(title="Weather skill · Brier by station × lead (model vs market)")
        for col in ("station", "lead", "n", "model", "market", "skill"):
            table.add_column(col)
        for b in buckets:
            table.add_row(
                b["station"], b["lead"], str(b["n"]), f"{b['brier_model']:.3f}",
                f"{b['brier_market']:.3f}" if b["brier_market"] is not None else "—",
                f"{b['skill']:+.0%}" if b["skill"] is not None else "—",
            )
        total = summarize_skill(buckets)
        if total:
            table.add_row("[bold]ALL[/bold]", "", str(total["n"]),
                          f"{total['brier_model']:.3f}", f"{total['brier_market']:.3f}",
                          f"{total['skill']:+.0%}" if total["skill"] is not None else "—")
        console.print(table)
        console.print("skill > 0 = beating the market price it traded against.")


@app.command()
def vital(out: str = typer.Option("vital.html", help="output HTML file")):
    """Generate a shareable performance card (like a Polymarket profile page)."""
    from .memory.journal import Journal
    from .report.vital import render_vital

    s = Settings.load()
    path = render_vital(Journal(s.db_path), s, out)
    console.print(f"[green]✓[/green] Wrote {path} — open it in a browser, screenshot, share.")


@app.command()
def hindcast(days: int = typer.Option(90, help="days of history to load (max 92)")):
    """Bulk-load leak-free forecast history so the baseline learns station
    bias/sigma immediately instead of over weeks of live settlements."""
    from .weather.hindcast import Hindcast
    from .weather.stations import KALSHI_SERIES, STATIONS
    from .weather.verification import VerificationStore

    s = Settings.load()
    store = VerificationStore(s.home / "weather-verification.jsonl")
    hc = Hindcast(store)
    stations = sorted({key for key, _ in KALSHI_SERIES.values()})
    for key in stations:
        station = STATIONS[key]
        try:
            g, st = hc.load_station(station, days)
            console.print(f"[green]✓[/green] {station.obs_id}: +{g} guidance, +{st} settlements")
        except Exception as e:
            console.print(f"[red]✗[/red] {station.obs_id}: {e}")

    table = Table(title="Baseline verification stats (bias / sigma / n)")
    table.add_column("station")
    for lead in range(1, 6):
        table.add_column(f"high L{lead}")
    for key in stations:
        row = [STATIONS[key].obs_id]
        for lead in range(1, 6):
            bias, sigma, n = store.stats(key, "high", lead)
            row.append(f"{bias:+.1f}/{sigma:.1f}/n={n}")
        table.add_row(*row)
    console.print(table)


@app.command()
def replay(
    days: int = typer.Option(21, help="settled days to replay"),
    min_edge: float = typer.Option(0.08, help="required edge after fees"),
):
    """Replay settled temperature markets against the statistical baseline
    alone (no LLM, no intraday obs) — a conservative expectancy bound."""
    from .weather.replay import replay_all, summarize
    from .weather.verification import VerificationStore

    s = Settings.load()
    store = VerificationStore(s.home / "weather-verification.jsonl")
    by_series = replay_all(store, days=days, min_edge=min_edge)

    table = Table(title=f"Baseline-only replay · last {days}d · min_edge={min_edge:.02f}")
    for col in ("series", "trades", "win rate", "pnl/contract", "total"):
        table.add_column(col)
    all_trades = []
    for series, trades in sorted(by_series.items()):
        all_trades += trades
        s_ = summarize(trades)
        if s_["n"]:
            table.add_row(series, str(s_["n"]), f"{s_['win_rate']:.0%}",
                          f"${s_['pnl_per_contract']:+.3f}", f"${s_['total_pnl']:+.2f}")
    total = summarize(all_trades)
    if total["n"]:
        table.add_row("[bold]ALL[/bold]", str(total["n"]), f"{total['win_rate']:.0%}",
                      f"${total['pnl_per_contract']:+.3f}", f"${total['total_pnl']:+.2f}")
        console.print(table)
        console.print(f"Priced edge kept after settlement: "
                      f"{total['pnl_per_contract'] / max(total['avg_edge_priced'], 1e-9):.0%} "
                      f"of the {total['avg_edge_priced']:.03f} average modeled edge.")
    else:
        console.print("No replay trades cleared the edge bar — run `openthomas hindcast` first?")


@app.command()
def improve(
    days: int = typer.Option(45, help="replay window for the gate"),
    dry_run: bool = typer.Option(False, "--dry-run", help="score candidates, write nothing"),
    history: bool = typer.Option(False, "--history", help="show the generation lineage"),
):
    """One self-improvement meta-cycle: mine the journal, propose parameter
    mutations, gate them on leak-free replay, promote or roll back. See
    docs/RSI.md — the trading loop also runs this daily after settlements."""
    from .improve.genome import GenerationStore
    from .improve.loop import Improver

    s = Settings.load()
    if history:
        table = Table(title="Generation lineage")
        for col in ("gen", "parent", "status", "proposer", "params", "note"):
            table.add_column(col)
        for g in GenerationStore(s.home).all():
            table.add_row(str(g.id), "—" if g.parent is None else str(g.parent),
                          g.status, g.proposer,
                          " ".join(f"{k.split('.')[-1]}={v}" for k, v in g.params.items()),
                          (g.rationale or g.note)[:60])
        console.print(table)
        return

    report = Improver(s).meta_cycle(days=days, dry_run=dry_run)
    console.print(f"Replay rows: {report.rows}")
    if report.rollback:
        console.print(f"[yellow]ROLLBACK[/yellow] {report.rollback}")
    for c in report.candidates:
        mark = "[green]✓[/green]" if c["verdict"] == "pass" else "[dim]✗[/dim]"
        params = " ".join(f"{k.split('.')[-1]}={v}" for k, v in c["params"].items())
        console.print(f"{mark} [{c['proposer']}] {params} · "
                      f"in ${c['held_in'].get('total_pnl', 0):+.2f} / "
                      f"out ${c['held_out'].get('total_pnl', 0):+.2f}"
                      + ("" if c["verdict"] == "pass" else f" · {c['verdict']}"))
    if report.promoted is not None:
        console.print(f"[green bold]Promoted generation {report.promoted}[/green bold] — {report.reason}")
    else:
        console.print(report.reason or "champion holds")
    if dry_run:
        console.print("[dim](dry run: nothing written)[/dim]")


@app.command()
def version():
    console.print(f"openthomas {__version__}")


if __name__ == "__main__":
    app()
