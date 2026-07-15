/* Renders feed.json. No framework, no build step: the page must stay readable
   by anyone auditing what the agent claimed and when. */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const money = (v) => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const signed = (v) => (v < 0 ? "−" : "+") + money(Math.abs(v));
const pct = (v) => (v < 0 ? "−" : "+") + (Math.abs(v) * 100).toFixed(2) + "%";
const cents = (p) => Math.round(p * 100) + "¢";
const nfmt = (n) => n.toLocaleString("en-US");

/** 18.4M / 940k / 812 — token counts get big; the exact digit never matters. */
function tokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + "M";
  if (n >= 1e3) return Math.round(n / 1e3) + "k";
  return String(n);
}

const plural = (n, one, many = one + "s") => `${nfmt(n)} ${n === 1 ? one : many}`;

const UTC = { timeZone: "UTC", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false };
const stamp = (iso) => new Date(iso).toLocaleString("en-US", UTC) + "Z";

function ago(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 2) return "just now";
  if (mins < 60) return mins + " min ago";
  const hrs = Math.round(mins / 60);
  if (hrs < 36) return hrs + "h ago";
  return Math.round(hrs / 24) + "d ago";
}

/* --- the disagreement scale ------------------------------------------------
   Two ticks on a 0–100¢ rule: what the market charges, what the agent believes.
   The span between them is the edge. Everything else on this page is context. */
function scale(t, { axis = false } = {}) {
  const mkt = t.p_market * 100;
  const mdl = t.p_model * 100;
  const wrap = el("div", "scale pre");
  wrap.dataset.dir = mdl >= mkt ? "up" : "down";
  wrap.style.setProperty("--market", mkt.toFixed(2));
  wrap.style.setProperty("--model", mdl.toFixed(2));
  wrap.style.setProperty("--lo", Math.min(mkt, mdl).toFixed(2));
  wrap.style.setProperty("--hi", Math.max(mkt, mdl).toFixed(2));
  wrap.setAttribute("role", "img");
  wrap.setAttribute("aria-label",
    `Market ${cents(t.p_market)}, model ${cents(t.p_model)}, edge ${cents(Math.abs(t.edge))} ` +
    `on the ${t.side.toUpperCase()} side.`);

  const rule = el("div", "scale-rule");
  for (const [k, v] of [["market", mkt], ["model", mdl]]) {
    const tick = el("div", `tick tick-${k}`);
    const label = el("span", "tick-label", `${k} ${cents(v / 100)}`);
    // A label near the right edge would run off the rule; hang it the other way.
    if (v > 82) label.dataset.anchor = "end";
    tick.append(label);
    rule.append(tick);
  }
  rule.append(el("div", "scale-span"));
  wrap.append(rule);

  if (axis) {
    const ax = el("div", "scale-axis");
    for (const n of [0, 25, 50, 75, 100]) ax.append(el("span", null, n + "¢"));
    wrap.append(ax);
  }

  const read = el("div", "scale-read");
  const edge = el("div", "scale-edge", cents(Math.abs(t.edge)));
  edge.append(el("small", null, "edge"));
  read.append(edge, el("span", "scale-side", `buy ${t.side}`));
  wrap.append(read);

  requestAnimationFrame(() => requestAnimationFrame(() => wrap.classList.remove("pre")));
  return wrap;
}

/* --- hero ------------------------------------------------------------------- */
function renderHero(feed) {
  const slot = $("#hero-slot");
  slot.textContent = "";
  const top = feed.theses.find((t) => t.edge !== null);
  if (!top) {
    slot.append(el("p", "empty",
      "No open claim clears the edge bar right now. The agent is holding cash — " +
      "which is what it is supposed to do when the market is priced correctly."));
    return;
  }

  const meta = el("p", "hero-meta");
  meta.append(el("span", "chip", top.status === "held" ? "position open" : "no position yet"));
  meta.append(el("span", null, top.platform));
  meta.append(el("span", null, stamp(top.ts)));
  slot.append(meta, el("h2", "hero-q", top.question));

  const s = scale(top, { axis: true });
  s.classList.add("hero-scale");
  slot.append(s);

  const claim = el("div", "claim");
  for (const [k, label, text] of [
    ["why", "Why", top.why],
    ["wrong", "Wrong if", top.invalidation],
  ]) {
    if (!text) continue;
    const row = el("div", "claim-row");
    row.dataset.k = k;
    row.append(el("div", "claim-k", label), el("p", "claim-v", text));
    claim.append(row);
  }
  slot.append(claim);
}


// Wins green, losses red — the same read as a Polymarket profile. Neutral facts
// (a drawdown, a Brier score) stay uncolored; green is reserved for money made.
const tone = (v) => (v > 0 ? "up" : v < 0 ? "down" : null);

function renderStats(feed) {
  const p = feed.performance;
  const rows = [
    ["Total P&L", signed(p.total_pnl), tone(p.total_pnl)],
    ["Positions value", money(p.positions_value), null],
    ["Realized P&L", signed(p.realized_pnl), tone(p.realized_pnl)],
    ["Unrealized P&L", signed(p.unrealized_pnl), tone(p.unrealized_pnl)],
    ["Biggest win", p.biggest_win === null ? "—" : signed(p.biggest_win),
     p.biggest_win > 0 ? "up" : null],
    ["Win rate", p.settled_trades
      ? `${(p.win_rate * 100).toFixed(0)}%<small> of ${p.settled_trades}</small>` : "—", null],
    ["Brier score", p.brier === null ? "—" : p.brier.toFixed(3), null],
    // Drawdown is a fact, not a loss — every strategy has one. Red is reserved
    // for money actually given back.
    ["Max drawdown", (p.max_drawdown * 100).toFixed(2) + "%", null],
  ];
  const dl = $("#stats-slot");
  dl.textContent = "";
  for (const [k, v, cls] of rows) {
    const box = el("div", "stat");
    const dd = el("dd", cls);
    dd.innerHTML = v;
    box.append(el("dt", null, k), dd);
    dl.append(box);
  }
}

/* --- home: the two headline trends + a compact stat block -------------------
   A P&L curve against a breakeven baseline and a positions-value curve, both
   straight off the recorded cycles — the pair a Polymarket profile leads with. */
function sparkline(sel, series, opts = {}) {
  const host = $(sel);
  if (!host) return;
  host.textContent = "";
  if (!series || series.length < 2) {
    host.append(el("p", "hchart-empty", "Two cycles needed for a trend."));
    return;
  }
  const W = 320, H = 96, PAD = 5, ns = "http://www.w3.org/2000/svg";
  const vals = series.map((c) => c[1]);
  const base = opts.baseline;
  const all = base != null ? vals.concat([base]) : vals;
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const x = (i) => (i / (series.length - 1)) * W;
  const y = (v) => PAD + (1 - (v - lo) / span) * (H - 2 * PAD);
  const last = vals[vals.length - 1];
  const down = base != null ? last < base : last < vals[0];
  const pts = series.map((c, i) => `${x(i).toFixed(1)},${y(c[1]).toFixed(1)}`).join(" ");

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "spark" + (down ? " down" : ""));
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const area = document.createElementNS(ns, "polygon");
  area.setAttribute("class", "spark-area");
  area.setAttribute("points", `0,${H} ${pts} ${W},${H}`);
  svg.append(area);
  if (base != null) {
    const bl = document.createElementNS(ns, "line");
    bl.setAttribute("class", "spark-base");
    bl.setAttribute("x1", 0); bl.setAttribute("x2", W);
    bl.setAttribute("y1", y(base).toFixed(1)); bl.setAttribute("y2", y(base).toFixed(1));
    svg.append(bl);
  }
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("class", "spark-line");
  line.setAttribute("points", pts);
  svg.append(line);
  host.append(svg);
}

function renderHomeCharts(feed) {
  const p = feed.performance;
  sparkline("#pnl-chart", p.pnl_curve, { baseline: 0 });
  sparkline("#posval-chart", p.positions_curve, {});
  const pnlNow = p.pnl_curve.length ? p.pnl_curve[p.pnl_curve.length - 1][1] : 0;
  const posNow = p.positions_curve.length ? p.positions_curve[p.positions_curve.length - 1][1] : 0;
  const set = (k, v, cls) => {
    const n = $(`[data-f="${k}"]`); if (!n) return;
    n.textContent = v; n.classList.remove("up", "down"); if (cls) n.classList.add(cls);
  };
  set("chart.pnl", signed(pnlNow), tone(pnlNow));
  set("chart.posval", money(posNow), null);
}

function renderHomeStats(feed) {
  const p = feed.performance;
  const rows = [
    ["Total P&L", signed(p.total_pnl), tone(p.total_pnl)],
    ["Positions value", money(p.positions_value), null],
    ["Win rate", p.settled_trades ? `${(p.win_rate * 100).toFixed(0)}%` : "—", null],
    ["Biggest win", p.biggest_win == null ? "—" : signed(p.biggest_win), p.biggest_win > 0 ? "up" : null],
  ];
  const dl = $("#home-stats");
  if (!dl) return;
  dl.textContent = "";
  for (const [k, v, cls] of rows) {
    const box = el("div", "stat");
    const dd = el("dd", cls);
    dd.innerHTML = v;
    box.append(el("dt", null, k), dd);
    dl.append(box);
  }
}

/* --- open claims ------------------------------------------------------------ */
function renderTheses(feed) {
  const slot = $("#theses-slot");
  slot.textContent = "";
  const list = feed.theses.filter((t) => t.edge !== null);
  $('[data-f="theses.count"]').textContent = list.length || "";

  if (!list.length) {
    slot.classList.remove("cards");
    slot.append(el("p", "empty", "No open claims. Every market the agent scanned this cycle was priced inside its edge bar."));
    return;
  }

  for (const t of list) {
    const card = el("article", "card");
    const head = el("div", "card-head");
    const chip = el("span", "chip", t.status === "held" ? "held" : "pending");
    chip.dataset.s = t.status;
    head.append(chip, el("span", null, t.platform), el("span", "card-ts", stamp(t.ts)));

    const s = scale(t);
    s.classList.add("card-scale");
    card.append(head, el("h3", "card-q", t.question), s);

    if (t.why) card.append(el("p", "card-why", t.why));
    if (t.invalidation) {
      const w = el("p", "card-wrong");
      w.append(el("b", null, "Wrong if"), document.createTextNode(t.invalidation));
      card.append(w);
    }
    slot.append(card);
  }
}

/* --- positions & settlements ------------------------------------------------- */
function renderRows(slotSel, countSel, items, empty, row) {
  const slot = $(slotSel);
  slot.textContent = "";
  $(countSel).textContent = items.length || "";
  if (!items.length) { slot.append(el("p", "empty", empty)); return; }
  const rows = el("div", "rows");
  for (const it of items) rows.append(row(it));
  slot.append(rows);
}

// A small subtitle over a table: the column's one-line total, colored by sign.
function slotNote(sel, text, v) {
  const n = $(sel);
  n.textContent = text || "";
  n.classList.remove("up", "down");
  if (text && v !== undefined && v !== 0) n.classList.add(v > 0 ? "up" : "down");
}

function renderPositions(feed) {
  const p = feed.performance;
  slotNote('[data-f="positions.value"]',
    feed.positions.length
      ? `${money(p.positions_value)} at market · ${signed(p.unrealized_pnl)} unrealized`
      : "", p.unrealized_pnl);
  renderRows("#positions-slot", '[data-f="positions.count"]', feed.positions,
    "No open positions. The agent is flat.", (q) => {
      const r = el("div", "row");
      const sub = el("div", "row-sub");
      sub.append(el("span", tone(q.unrealized), signed(q.unrealized)),
                 document.createTextNode(` · ${q.side} ${nfmt(q.qty)} @ ${cents(q.avg_cost)}`));
      r.append(el("div", "row-q", q.question),
               el("div", "row-n", money(q.value)), sub,
               el("div", "row-side", q.platform));
      return r;
    });
}

function renderTrack(feed) {
  const p = feed.performance;
  slotNote('[data-f="track.pnl"]',
    feed.track_record.length ? `${signed(p.realized_pnl)} realized · ${p.settled_trades} settled` : "",
    p.realized_pnl);
  renderRows("#track-slot", '[data-f="track.count"]', feed.track_record,
    "Nothing has settled yet. The track record starts at the first resolution.", (s) => {
      const r = el("div", "row");
      r.append(el("div", "row-q", s.question),
               el("div", "row-n " + (tone(s.pnl) || ""), signed(s.pnl)),
               el("div", "row-sub", `resolved ${s.outcome} · ${s.ts.slice(0, 10)}`),
               el("div", "row-side", s.platform));
      return r;
    });
}

/* --- activity: the tape ------------------------------------------------------
   Every entry, exit, and resolution in one stream — the Polymarket "Activity"
   tab. Buys/sells carry the price paid; resolutions carry the realized PnL. */
function renderActivity(feed) {
  const items = feed.activity || [];
  const slot = $("#activity-slot");
  $('[data-f="activity.count"]').textContent = items.length || "";
  slot.textContent = "";
  if (!items.length) {
    slot.append(el("p", "empty", "No activity yet. The tape starts at the first fill."));
    return;
  }
  const rows = el("div", "rows");
  for (const a of items) {
    const settle = a.kind === "settle";
    const r = el("div", "row");
    const q = el("div", "row-q");
    const chip = el("span", "act-kind");
    chip.dataset.k = a.kind;
    chip.textContent = settle ? "resolved" : a.kind;
    q.append(chip, document.createTextNode(" " + a.question));
    const n = settle
      ? el("div", "row-n " + (tone(a.pnl) || ""), signed(a.pnl))
      : el("div", "row-n", money(a.cost));
    const sub = settle
      ? `${a.outcome.toUpperCase()} · ${a.ts.slice(0, 10)}`
      : `${a.side} ${nfmt(a.qty)} @ ${cents(a.price)} · ${a.ts.slice(0, 10)}`;
    r.append(q, n, el("div", "row-sub", sub), el("div", "row-side", a.platform));
    rows.append(r);
  }
  slot.append(rows);
}

/* --- field notes: the daily dispatch blog -----------------------------------
   One entry per day, newest first: a short prose read of the day plus the exact
   line to copy to X. The report is templated on the trading box; this only
   renders it and lends a copy button. */
function copyForX(text, btn) {
  const flash = () => { const was = btn.textContent; btn.textContent = "Copied"; btn.classList.add("ok");
    setTimeout(() => { btn.textContent = was; btn.classList.remove("ok"); }, 1400); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(flash).catch(() => legacyCopy(text, flash));
  } else { legacyCopy(text, flash); }
}
function legacyCopy(text, done) {
  const t = el("textarea"); t.value = text; t.setAttribute("readonly", "");
  t.style.position = "fixed"; t.style.opacity = "0"; document.body.append(t);
  t.select(); try { document.execCommand("copy"); } catch (e) { /* clipboard blocked */ }
  t.remove(); done();
}
function renderNotes(feed) {
  const items = (feed.reports || []).slice(0, 7);
  const slot = $("#notes-slot");
  $('[data-f="notes.count"]').textContent = items.length || "";
  slot.textContent = "";
  if (!items.length) {
    slot.append(el("p", "empty", "The first dispatch posts on the next cycle."));
    return;
  }
  for (const r of items) {
    const note = el("article", "note");
    const head = el("div", "note-head");
    head.append(el("span", "note-date", r.date), el("h3", "note-title", r.title));
    note.append(head);
    for (const para of r.body || []) note.append(el("p", "note-body", para));

    const box = el("div", "note-x");
    box.append(el("span", "note-x-lab", "for X"), el("p", "note-x-text", r.tweet));
    const btn = el("button", "note-copy", "Copy for X");
    btn.type = "button";
    btn.addEventListener("click", () => copyForX(r.tweet, btn));
    box.append(btn);
    note.append(box);
    slot.append(note);
  }
}

/* --- self-improvement -------------------------------------------------------- */
function renderRsi(feed) {
  const slot = $("#rsi-slot");
  slot.textContent = "";
  const { active_generation: g, meta_cycles: cycles } = feed.rsi;

  if (!g) {
    slot.append(el("p", "empty",
      "No generation has cleared the promotion gate yet, so the operator's config is still in force. " +
      "That is the safe default: a change ships only on evidence."));
  } else {
    const box = el("div", "gen");
    const head = el("div", "gen-head");
    head.append(el("span", "gen-id", "Generation " + g.id),
                el("span", null, g.operator + " operator"),
                el("span", null, "proposed by " + g.proposer),
                el("span", null, "promoted " + g.created.slice(0, 10)));
    box.append(head);
    if (g.rationale) box.append(el("p", "gen-why", g.rationale));
    const params = el("div", "params");
    for (const [k, v] of Object.entries(g.params)) {
      const chip = el("span", "param");
      chip.append(document.createTextNode(k.split(".").pop() + " "), el("b", null, String(v)));
      params.append(chip);
    }
    box.append(params);
    slot.append(box);
  }

  if (!cycles.length) {
    slot.append(el("p", "empty", "The evolution loop has not run a meta-cycle yet."));
    return;
  }
  for (const c of cycles) {
    const row = el("div", "cycle");
    const body = el("div", "cycle-body");
    const passed = c.candidates.filter((x) => x.verdict === "pass").length;

    body.append(el("b", null, c.operator + " operator"), document.createTextNode(
      ` · ${plural(c.replay_rows, "replay row")} · ${plural(c.candidates.length, "candidate")}, `));
    body.append(el("span", passed ? "verdict-pass" : null, `${passed} cleared the gate`));
    if (c.rollback) body.append(document.createTextNode(" · rolled back: " + c.rollback));
    if (c.reason) body.append(el("div", null, c.reason));
    row.append(el("div", "cycle-ts", stamp(c.ts)), body);
    slot.append(row);
  }
}

/* --- compute ----------------------------------------------------------------- */
function tokenColumn(kind, title, dot, b, extra) {
  const col = el("div", "tok-col" + (kind ? " " + kind : ""));
  const h = el("h4");
  if (dot) h.append(el("span", "field-dot"));
  h.append(document.createTextNode(title));
  const big = el("div", "tok-big");
  big.append(document.createTextNode(tokens(b.total_tokens)),
             el("span", "tok-unit", "tokens"));
  const meta = el("div", "tok-meta");
  const put = (label, val) => {
    const s = el("span");
    s.append(document.createTextNode(label + " "), el("b", null, val));
    meta.append(s);
  };
  put("in", tokens(b.prompt_tokens));
  put("out", tokens(b.completion_tokens));
  put("calls", nfmt(b.calls));
  if (extra) meta.append(el("span", null, extra));
  col.append(h, big, meta);
  return col;
}

function renderCompute(feed) {
  const slot = $("#compute-slot");
  const c = feed.compute;
  const session = c.session || { total: c.total, since: null };
  slot.textContent = "";

  const split = el("div", "tok-split");
  split.append(tokenColumn("", "All-time", false, c.total));
  split.append(tokenColumn("run", "This run", true, session.total,
    session.since ? "since " + ago(session.since) : "since the loop last started"));
  slot.append(split);

  if (!c.ledger_started) {
    slot.append(el("p", "empty",
      "Token accounting starts with the next cycle. The forecasts on record predate the ledger, " +
      "so their cost is not reconstructable — and this page does not guess."));
    return;
  }

  const max = Math.max(...c.by_node.map((n) => n.total_tokens), 1);
  const bars = el("div", "bars");
  for (const n of c.by_node) {
    const row = el("div", "bar-row");
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = (n.total_tokens / max) * 100 + "%";
    track.append(fill);
    row.append(el("div", "bar-k", n.node), track,
               el("div", "bar-v", `${tokens(n.total_tokens)} · ${plural(n.calls, "call")}`));
    bars.append(row);
  }
  slot.append(bars);

  const flat = c.total.calls_without_usage;
  const note = `All-time counted since ${c.ledger_started.slice(0, 10)}; the split above is per model role.` + (flat
    ? ` ${plural(flat, "call")} went to a flat-rate subscription endpoint that reports no token counts; they are not in the totals above.`
    : "");
  slot.append(el("p", "empty", note));
}

/* --- liveness: the strip and the field track a running agent ----------------
   Live if the loop's last cycle is recent relative to its own cadence. The
   timestamp is re-read every second so "updated 3s ago" counts up on screen —
   a still page reads as a dead one. */
let LIVE = { lastCycle: null, cycleMax: 20 };

function renderLive(feed) {
  const st = feed.status || {};
  LIVE.lastCycle = st.last_cycle || feed.performance.as_of || null;
  LIVE.cycleMax = (feed.agent.cycle_minutes || 20) * 2 + 10;

  const set = (k, v) => { const n = $(`[data-f="${k}"]`); if (n) n.textContent = v; };
  set("status.cycle", st.cycles_this_run ? `cycle ${nfmt(st.cycles_this_run)}` : "warming up");
  tickLive();
}

/* --- the planet: the whole board, plus click-to-expand ---------------------- */
const STATE_LABEL = { market: "on the board", pending: "edge found",
                      held: "bet placed", won: "won", lost: "lost" };

const PRIO = { held: 4, pending: 3, won: 2, lost: 1, market: 0 };
let GROUPS = {};   // place → { lat, lon, place, items[], primary, state, count, weight }

function groupBoard(feed) {
  const rows = (feed.board && feed.board.markets) || [];
  const g = {};
  for (const e of rows) {
    if (!e.loc) continue;
    const k = e.place || `${e.loc.lat},${e.loc.lon}`;
    (g[k] || (g[k] = { lat: e.loc.lat, lon: e.loc.lon, place: e.place || k, items: [] })).items.push(e);
  }
  for (const k in g) {
    const grp = g[k];
    grp.items.sort((a, b) => PRIO[b.state] - PRIO[a.state] || Math.abs(b.edge || 0) - Math.abs(a.edge || 0));
    grp.primary = grp.items[0];
    grp.state = grp.primary.state;
    grp.count = grp.items.length;
    grp.weight = Math.min(Math.abs(grp.primary.edge || 0) / 0.3, 1);
  }
  return g;
}

function renderGlobe(feed) {
  GROUPS = groupBoard(feed);
  const markers = Object.values(GROUPS).map((g) => ({
    lat: g.lat, lon: g.lon, place: g.place, state: g.state, weight: g.weight, count: g.count,
  }));
  window.OTGlobe && OTGlobe.setMarkers(markers);
  window.OTGlobe && OTGlobe.setSkill(feed.skill || []);
  window.OTGlobe && OTGlobe.setAnomaly(feed.anomaly || []);
  window.OTGlobe && OTGlobe.setTemps(feed.temperature || null);

  const total = ((feed.board && feed.board.markets) || []).length;
  const ours = markers.filter((m) => m.state !== "market").length;
  const cap = $('[data-f="globe.caption"]');
  if (cap) cap.textContent = total
    ? `${plural(total, "market")} · ${plural(ours, "live edge")} · ${plural(markers.length, "city", "cities")}`
    : "no markets yet";

  const tsrc = $('[data-f="temperature.source"]');
  const legend = document.querySelector(".templegend");
  if (feed.temperature) {
    const t = feed.temperature, src = t.source || "Open-Meteo";
    const kind = /graphcast|gencast/i.test(src) ? "our forecast" : "current temp";
    if (tsrc) tsrc.textContent = `${kind}${t.as_of ? " · valid " + ago(t.as_of) : ""} · ${src}`;
    if (legend) legend.style.display = "";
  } else if (legend) { legend.style.display = "none"; }
}

/* --- the edge lens: where the money is right now --------------------------- */
function oppScore(m) {
  const edge = Math.abs(m.edge || 0);
  const liq = m.volume == null ? 0.5 : Math.min(m.volume / 50000, 1);
  let urg = 0.6;
  if (m.close) { const days = (new Date(m.close) - Date.now()) / 86400000;
    urg = days <= 0 ? 0.2 : days < 2 ? 1 : days < 5 ? 0.85 : days < 10 ? 0.6 : 0.4; }
  return edge * (0.5 + 0.5 * liq) * (0.5 + 0.5 * urg);
}
function closesIn(iso) {
  const h = (new Date(iso) - Date.now()) / 3600000;
  return h < 0 ? "now" : h < 24 ? Math.round(h) + "h" : Math.round(h / 24) + "d";
}

function railHead(title, foot) {
  const t = document.querySelector(".rail-title"); if (t) t.textContent = title;
  const f = document.querySelector(".opprail-foot"); if (f) f.textContent = foot;
}

function renderOpps(feed) {
  const rows = ((feed.board && feed.board.markets) || [])
    .filter((m) => m.loc && (m.state === "pending" || m.state === "held") && m.edge != null)
    .sort((a, b) => oppScore(b) - oppScore(a)).slice(0, 14);
  railHead("Where the edges are", "Ranked by edge × liquidity × time to close. Click to fly there.");
  const slot = $("#opprail-list");
  const n = $('[data-f="opp.count"]');
  if (n) n.textContent = rows.length || "";
  if (!slot) return;
  slot.textContent = "";
  if (!rows.length) {
    slot.append(el("p", "opp-empty",
      "No live edge clears the bar right now — the agent is holding cash, which is the point."));
    return;
  }
  for (const m of rows) {
    const r = el("button", "opp"); r.dataset.s = m.state;
    const head = el("div", "opp-head");
    head.append(el("span", "opp-place", m.place),
                el("span", "opp-edge", cents(Math.abs(m.edge)) + " edge"));
    const meta = el("div", "opp-meta");
    meta.append(el("span", "opp-side", m.state === "held" ? "holding" : (m.side ? `buy ${m.side}` : "")),
                el("span", null, `us ${(m.p_model * 100).toFixed(0)}% · mkt ${m.mid != null ? (m.mid * 100).toFixed(0) + "%" : "—"}`));
    if (m.close) meta.append(el("span", "opp-close", "closes " + closesIn(m.close)));
    r.append(head, el("div", "opp-q", m.question), meta);
    r.addEventListener("mouseenter", () => window.OTGlobe && OTGlobe.highlight(m.place));
    r.addEventListener("mouseleave", () => window.OTGlobe && OTGlobe.highlight(null));
    r.addEventListener("click", () => {
      window.OTGlobe && OTGlobe.focus(m.loc.lon, m.loc.lat);
      openDetail({ place: m.place });
    });
    slot.append(r);
  }
}

function renderSkill(feed) {
  const rows = (feed.skill || []).filter((s) => s.disagreement != null).slice(0, 16);
  railHead("Where we fight the crowd",
    "Size = how contrarian we are; ring = the settled verdict vs the market. Click to fly there.");
  const slot = $("#opprail-list");
  const n = $('[data-f="opp.count"]');
  if (n) n.textContent = rows.length || "";
  if (!slot) return;
  slot.textContent = "";
  if (!rows.length) { slot.append(el("p", "opp-empty", "No forecasts on record yet.")); return; }
  for (const s of rows) {
    const r = el("button", "opp");
    const head = el("div", "opp-head");
    head.append(el("span", "opp-place", s.place), el("span", "opp-edge", cents(s.disagreement) + " apart"));
    const meta = el("div", "opp-meta");
    if (s.n_settled) {
      const won = s.pnl >= 0;
      meta.append(el("span", won ? "opp-win" : "opp-lose", won ? "beats market" : "behind market"),
                  el("span", null, `${Math.round((s.win_rate || 0) * 100)}% win · ${signed(s.pnl)} · ${s.n_settled} settled`));
    } else {
      meta.append(el("span", "opp-side", `${s.n_forecasts} forecasts · unsettled`));
    }
    r.append(head, meta);
    r.addEventListener("mouseenter", () => window.OTGlobe && OTGlobe.highlight(s.place));
    r.addEventListener("mouseleave", () => window.OTGlobe && OTGlobe.highlight(null));
    r.addEventListener("click", () => { window.OTGlobe && OTGlobe.focus(s.lon, s.lat); openDetail({ place: s.place }); });
    slot.append(r);
  }
}

function renderAnomaly(feed) {
  const rows = (feed.anomaly || []).slice(0, 16);
  railHead("What's unusual right now",
    "Departure from the monthly normal (°C). Red = hotter than usual, blue = colder — where the market may lag.");
  const slot = $("#opprail-list");
  const n = $('[data-f="opp.count"]');
  if (n) n.textContent = rows.length || "";
  if (!slot) return;
  slot.textContent = "";
  if (!rows.length) { slot.append(el("p", "opp-empty", "Anomaly needs the temperature field and city normals — loading.")); return; }
  for (const a of rows) {
    const hot = a.anomaly >= 0;
    const r = el("button", "opp");
    const head = el("div", "opp-head");
    head.append(el("span", "opp-place", a.place),
                el("span", hot ? "opp-lose" : "opp-side", `${hot ? "+" : ""}${a.anomaly}°C`));
    const meta = el("div", "opp-meta");
    meta.append(el("span", null, `now ${a.temp}° · normal ${a.normal}°`),
                el("span", null, hot ? "hotter than usual" : "colder than usual"));
    r.append(head, meta);
    r.addEventListener("mouseenter", () => window.OTGlobe && OTGlobe.highlight(a.place));
    r.addEventListener("mouseleave", () => window.OTGlobe && OTGlobe.highlight(null));
    r.addEventListener("click", () => { window.OTGlobe && OTGlobe.focus(a.lon, a.lat); openDetail({ place: a.place }); });
    slot.append(r);
  }
}

let CURRENT_FEED = null, CURRENT_LENS = "temp";
const RAILS = { edge: renderOpps, skill: renderSkill, anomaly: renderAnomaly };
function refreshRail() {
  const rail = $("#opprail"); if (!rail) return;
  const render = RAILS[CURRENT_LENS];
  if (!CURRENT_FEED || !render) { rail.hidden = true; return; }
  rail.hidden = false;
  render(CURRENT_FEED);
}
function setLens(mode) {
  CURRENT_LENS = mode;
  document.querySelectorAll(".lens").forEach((b) => b.classList.toggle("on", b.dataset.lens === mode));
  window.OTGlobe && OTGlobe.setLens(mode);
  refreshRail();
  updateTimeaxis();
}

/* --- the time axis: scrub our GraphCast forecast forward, day by day -------- */
let SERIES = null, PLAYING = null;

async function loadSeries() {
  try {
    const r = await fetch("tempseries.json", { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    if (!s.leads || !s.leads.length) return;
    SERIES = s;
    const sl = $("#ta-slider");
    if (sl) { sl.max = s.leads.length - 1; sl.value = 0; sl.addEventListener("input", () => setLead(+sl.value)); }
    const play = $("#ta-play"); if (play) play.addEventListener("click", togglePlay);
    updateTimeaxis();
    setLead(0);
  } catch (e) { /* no series → no time axis */ }
}

function leadGridFor(i) {
  if (i <= 0) return null;   // "now" uses the crisp default field
  const L = SERIES.leads[i];
  return { nx: SERIES.nx, ny: SERIES.ny, lat0: SERIES.lat0, lon0: SERIES.lon0,
           dlat: SERIES.dlat, dlon: SERIES.dlon, temps: L.temps };
}

function setLead(i) {
  if (!SERIES) return;
  const sl = $("#ta-slider"); if (sl && +sl.value !== i) sl.value = i;
  window.OTGlobe && OTGlobe.showLead(leadGridFor(i));
  const L = SERIES.leads[i], lab = $("#ta-label");
  if (lab) lab.textContent = i === 0 ? "now"
    : `+${Math.round(L.lead_h / 24)}d · ${new Date(L.as_of).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" })}`;
}

function togglePlay() {
  const btn = $("#ta-play");
  if (PLAYING) { clearInterval(PLAYING); PLAYING = null; if (btn) btn.textContent = "▶"; return; }
  if (btn) btn.textContent = "❚❚";
  PLAYING = setInterval(() => {
    const sl = $("#ta-slider"); if (!sl) return;
    setLead(+sl.value >= +sl.max ? 0 : +sl.value + 1);
  }, 950);
}

function updateTimeaxis() {
  const ta = $("#timeaxis"); if (!ta) return;
  ta.hidden = !(SERIES && CURRENT_LENS === "temp");
  if (ta.hidden && PLAYING) { clearInterval(PLAYING); PLAYING = null; const b = $("#ta-play"); if (b) b.textContent = "▶"; }
}

function placeTip(tip) {  // shared positioning
  tip.hidden = false;
  const band = tip.parentElement.getBoundingClientRect();
  tip.style.left = Math.max(10, Math.min(tip._x + 16, band.width - 300)) + "px";
  tip.style.top = Math.max(10, Math.min(tip._y - tip.offsetHeight / 2, band.height - tip.offsetHeight - 12)) + "px";
}

function globeTip(m, x, y) {
  const tip = $("#globe-tip");
  if (!tip) return;
  if (!m) { tip.hidden = true; return; }
  tip.textContent = ""; tip._x = x; tip._y = y;

  if (m.anomaly !== undefined && m.normal !== undefined) {   // anomaly-lens marker
    const hot = m.anomaly >= 0;
    tip.append(el("b", null, m.place),
      el("span", "tip-detail", `${hot ? "+" : ""}${m.anomaly}°C ${hot ? "hotter" : "colder"} than normal`),
      el("span", "tip-q", `${m.estimated ? "est. " : ""}today ~${m.temp}°C · normal for the month ${m.normal}°C`),
      el("span", "tip-more", "click for detail"));
    placeTip(tip); return;
  }

  if (m.disagreement !== undefined) {   // skill-lens marker
    tip.append(el("b", null, m.place),
      el("span", "tip-detail", `${cents(m.disagreement)} apart from the market · ${m.n_forecasts} forecasts`));
    tip.append(el("span", "tip-q", m.n_settled
      ? `settled ${m.n_settled}: ${m.pnl >= 0 ? "beating" : "behind"} the market · ${Math.round((m.win_rate || 0) * 100)}% win`
        + (m.brier_market != null ? ` · Brier ${m.brier_model} vs ${m.brier_market}` : "")
      : "not enough settled to score yet"));
    tip.append(el("span", "tip-more", "click for detail"));
    placeTip(tip); return;
  }

  const g = GROUPS[m.place] || { items: [m], primary: m };
  const p = g.primary;
  tip.append(el("b", null, m.place));
  const line = STATE_LABEL[p.state] + (p.mid != null ? ` · mkt ${cents(p.mid)}` : "") +
    (p.edge != null ? ` · edge ${cents(Math.abs(p.edge))}` : "");
  tip.append(el("span", "tip-detail", line), el("span", "tip-q", p.question));
  tip.append(el("span", "tip-more", g.items.length > 1
    ? `${g.items.length} markets here · click to see all` : "click for detail"));
  placeTip(tip);
}

/* --- click a place → all its markets, book + our analysis, history included -- */
function kv(k, v) { const w = el("div", "detail-kv"); w.append(el("dt", null, k), el("dd", null, v)); return w; }

function cityItem(m, open) {
  const it = el("details", "citem"); if (open) it.open = true;
  const sum = el("summary", "citem-head");
  const badge = el("span", "detail-badge", STATE_LABEL[m.state]); badge.dataset.s = m.state;
  sum.append(badge, el("span", "citem-q", m.question),
             el("span", "citem-mkt", m.mid != null ? cents(m.mid) : "—"));
  it.append(sum);

  const body = el("div", "citem-body");
  const book = el("div", "detail-book");
  if (m.yes_bid != null && m.yes_ask != null) book.append(kv("Bid / ask", `${cents(m.yes_bid)} / ${cents(m.yes_ask)}`));
  if (m.volume) book.append(kv("Volume 24h", money(m.volume)));
  if (m.close) book.append(kv("Closes", stamp(m.close)));
  if (book.children.length) body.append(book);

  if (m.state !== "market") {
    const our = el("div", "detail-our");
    const g = el("div", "detail-book");
    if (m.p_model != null) g.append(kv("Our P(YES)", (m.p_model * 100).toFixed(0) + "%"));
    if (m.edge != null) g.append(kv("Edge", cents(Math.abs(m.edge)) + (m.side ? ` · buy ${m.side}` : "")));
    if (m.outcome) { const dd = kv("Resolved", `${m.outcome.toUpperCase()} · ${signed(m.pnl || 0)}`);
      if ((m.pnl || 0) < 0) dd.querySelector("dd").classList.add("down"); g.append(dd); }
    if (g.children.length) our.append(g);
    if (m.why) our.append(el("p", "detail-why", m.why));
    if (m.reasoning) our.append(el("p", "detail-reason", m.reasoning));
    if (our.children.length) body.append(our);
  }
  it.append(body);
  return it;
}

function openDetail(marker) {
  const box = $("#detail");
  const g = GROUPS[marker && marker.place];
  if (!box || !g) return;
  box.textContent = "";
  const x = el("button", "detail-x", "✕"); x.setAttribute("aria-label", "Close");
  x.addEventListener("click", closeDetail);
  box.append(x, el("div", "detail-city", g.place));

  const live = g.items.filter((i) => i.state === "held" || i.state === "pending").length;
  const settled = g.items.filter((i) => i.state === "won" || i.state === "lost").length;
  const parts = [plural(g.items.length, "market")];
  if (live) parts.push(`${live} live`);
  if (settled) parts.push(`${settled} settled`);
  box.append(el("div", "detail-citysub", parts.join(" · ") + " on the venue"));

  const list = el("div", "detail-list");
  g.items.forEach((it, i) => list.append(cityItem(it, i === 0 && it.state !== "market")));
  box.append(list);
  box.hidden = false;
  requestAnimationFrame(() => box.classList.add("open"));
}
function closeDetail() { const b = $("#detail"); if (b) { b.classList.remove("open"); b.hidden = true; } }

function tickLive() {
  const stale = LIVE.lastCycle
    ? (Date.now() - new Date(LIVE.lastCycle)) / 60000 > LIVE.cycleMax
    : true;
  const live = $('[data-f="status.live"]');
  if (live) live.textContent = stale ? "idle" : "live";
  const upd = $('[data-f="status.updated"]');
  if (upd) upd.innerHTML = LIVE.lastCycle
    ? "updated " + ago(LIVE.lastCycle) : "no cycle yet";
  $("#home-status") && $("#home-status").classList.toggle("is-stale", stale);
  $("#globe") && $("#globe").classList.toggle("is-stale", stale);
}

/* --- chrome ------------------------------------------------------------------- */
function renderChrome(feed) {
  const p = feed.performance;
  const set = (k, v) => { const n = $(`[data-f="${k}"]`); if (n) n.textContent = v; };

  set("agent.mode_chip", `${feed.agent.mode} · ${feed.agent.focus}`);
  set("performance.as_of_rel", p.as_of ? "cycle " + ago(p.as_of) : "no cycles yet");
  set("performance.account_value", money(p.account_value));
  set("generated_at", stamp(feed.generated_at));

  // Token spend, next to the money — updated on every feed, all-time and this run.
  const c = feed.compute;
  const sess = (c.session && c.session.total) || c.total;
  set("compute.total_tokens", tokens(c.total.total_tokens));
  set("compute.session_tokens", tokens(sess.total_tokens));

  const delta = $('[data-f="performance.return_line"]');
  delta.textContent = "";
  const v = el("span", p.return_pct < 0 ? "down" : null, pct(p.return_pct));
  delta.append(v, document.createTextNode(` since ${money(p.bankroll)} start`));

  const nav = $("#links-slot");
  nav.textContent = "";
  const links = [
    ["Source on GitHub", feed.links.github],
    ["Models & data on Hugging Face", feed.links.huggingface],
    ["Claims, timestamped on X", feed.links.x],
  ];
  for (const [label, href] of links) {
    if (!href) continue;
    const a = el("a", null, label);
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener";
    nav.append(a);
  }
}

/* --- routing: one page at a time --------------------------------------------
   Hash routes so the static host needs no rewrite rules. Every view is rendered
   once per feed; the router only decides which one is shown. */
const ROUTES = ["home", "positions", "activity", "daily", "blog", "system"];
function currentRoute() {
  const raw = (location.hash || "").replace(/^#\/?/, "");
  const [top, sub] = raw.split("/");
  return { view: ROUTES.includes(top) ? top : "home", sub: sub || "" };
}
function showView() {
  const { view, sub } = currentRoute();
  document.querySelectorAll("#app > [data-view]").forEach((v) => { v.hidden = v.dataset.view !== view; });
  document.querySelectorAll(".nav-links a").forEach((a) => a.classList.toggle("on", a.dataset.route === view));
  document.body.classList.toggle("is-home", view === "home");
  if (view === "blog") renderBlogRoute(sub);
  window.scrollTo(0, 0);
}

/* --- blog: authored technical notes, loaded from blog.json ------------------- */
let BLOG = null;
async function loadBlog() {
  try { const r = await fetch("blog.json", { cache: "no-store" }); if (r.ok) BLOG = await r.json(); }
  catch (e) { /* no blog file → the page just shows an empty blog */ }
  const rt = currentRoute();
  if (rt.view === "blog") renderBlogRoute(rt.sub);
}
function blogMeta(post) {
  const meta = el("div", "blog-meta");
  meta.append(el("span", "blog-date", post.date));
  for (const t of post.tags || []) meta.append(el("span", "blog-tag", t));
  return meta;
}
function renderBlogRoute(slug) {
  const slot = $("#blog-slot");
  if (!slot) return;
  const posts = (BLOG && BLOG.posts) || [];
  slot.textContent = "";
  if (!posts.length) { slot.append(el("p", "empty", "The blog is warming up. Check back soon.")); return; }
  const post = slug && posts.find((p) => p.slug === slug);
  if (post) {
    const back = el("a", "blogback", "← All posts"); back.href = "#/blog";
    slot.append(back);
    const art = el("article", "blogpost");
    art.append(blogMeta(post), el("h1", "blogpost-title", post.title));
    for (const para of post.body || []) art.append(el("p", "blogpost-p", para));
    slot.append(art);
  } else {
    const head = el("header", "page-head");
    head.append(el("h1", null, "Blog"), el("p", null,
      "Technical notes behind the trades — an edge we think the crowd is missing, a temperature " +
      "signal worth watching, a station whose numbers look wrong."));
    slot.append(head);
    const list = el("div", "bloglist");
    for (const p of posts) {
      const a = el("a", "bloglink"); a.href = `#/blog/${p.slug}`;
      a.append(blogMeta(p), el("h2", "bloglink-title", p.title), el("p", "bloglink-sum", p.summary || ""));
      list.append(a);
    }
    slot.append(list);
  }
}

/* --- feed loading + the live loop -------------------------------------------
   The page reloads the feed on its own so it stays current without a manual
   refresh, and drives the 3D field from the agent's state: warmth follows the
   return, and a ripple fires whenever the cycle counter advances. */
let SEEN = { generated: null, cycles: null };

async function fetchFeed() {
  try {
    const resp = await fetch("feed.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(resp.status);
    return await resp.json();
  } catch (err) {
    return null;
  }
}

function applyFeed(feed) {
  renderChrome(feed);
  renderHomeCharts(feed);
  renderHomeStats(feed);
  renderHero(feed);
  renderStats(feed);
  renderTheses(feed);
  renderPositions(feed);
  renderTrack(feed);
  renderActivity(feed);
  renderNotes(feed);
  renderRsi(feed);
  renderCompute(feed);
  renderLive(feed);
  renderGlobe(feed);
  CURRENT_FEED = feed;
  refreshRail();
  SEEN.generated = feed.generated_at;
}

async function main() {
  window.OTStars && OTStars.init("stars");
  window.OTGlobe && OTGlobe.init("globe-canvas", {
    onHover: globeTip,
    onSelect: (m) => { openDetail(m); window.OTGlobe.focus(m.lon, m.lat); },
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
  document.querySelectorAll(".lens").forEach((b) => b.addEventListener("click", () => setLens(b.dataset.lens)));

  window.addEventListener("hashchange", showView);
  showView();                                  // route immediately, before the feed lands
  loadBlog();                                  // authored posts, from blog.json

  const feed = await fetchFeed();
  if (!feed) {
    const slot = $("#hero-slot");
    if (slot) { slot.textContent = ""; slot.append(el("p", "empty",
      "The feed did not load. The agent keeps trading either way — reload, or read feed.json directly.")); }
    return;
  }
  applyFeed(feed);
  loadSeries();                                // our GraphCast forecast series, if published

  setInterval(tickLive, 1000);                 // the "updated Ns ago" counter climbs live
  setInterval(async () => {                    // pull fresh data without a reload
    const f = await fetchFeed();
    if (f && f.generated_at !== SEEN.generated) applyFeed(f);
  }, 45000);
}

main();
