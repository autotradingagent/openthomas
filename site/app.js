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

/* --- equity ----------------------------------------------------------------- */
function renderCurve(feed) {
  const slot = $("#curve-slot");
  const curve = feed.performance.equity_curve;
  slot.textContent = "";
  if (curve.length < 2) {
    slot.append(el("p", "empty", "The equity curve needs two cycles. The first is on the clock."));
    return;
  }

  const W = 600, H = 132, PAD = 3;
  const vals = curve.map((c) => c[1]).concat([feed.performance.bankroll]);
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const x = (i) => (i / (curve.length - 1)) * W;
  const y = (v) => PAD + (1 - (v - lo) / span) * (H - 2 * PAD);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "curve");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label",
    `Account value from ${money(curve[0][1])} to ${money(curve[curve.length - 1][1])} over ${curve.length} cycles.`);

  const ns = "http://www.w3.org/2000/svg";
  const base = document.createElementNS(ns, "line");
  base.setAttribute("class", "curve-base");
  base.setAttribute("x1", 0); base.setAttribute("x2", W);
  base.setAttribute("y1", y(feed.performance.bankroll));
  base.setAttribute("y2", y(feed.performance.bankroll));
  svg.append(base);

  const line = document.createElementNS(ns, "polyline");
  const down = curve[curve.length - 1][1] < feed.performance.bankroll;
  line.setAttribute("class", "curve-line" + (down ? " down" : ""));
  line.setAttribute("points", curve.map((c, i) => `${x(i)},${y(c[1])}`).join(" "));
  svg.append(line);
  slot.append(svg);

  const cap = el("div", "curve-caption");
  cap.append(el("span", null, curve[0][0].slice(0, 10)),
             el("span", null, `${curve.length} cycles · dashed line = $${nfmt(feed.performance.bankroll)} start`),
             el("span", null, curve[curve.length - 1][0].slice(0, 10)));
  slot.append(cap);
}

function renderStats(feed) {
  const p = feed.performance;
  const rows = [
    ["Realized P&L", signed(p.realized_pnl), p.realized_pnl < 0],
    ["Settled", `${p.settled_trades}<small> trades</small>`, false],
    ["Win rate", p.settled_trades ? (p.win_rate * 100).toFixed(0) + "%" : "—", false],
    ["Brier score", p.brier === null ? "—" : p.brier.toFixed(3), false],
    // Drawdown is a fact, not a loss — every strategy has one. Red is reserved
    // for money actually given back.
    ["Max drawdown", (p.max_drawdown * 100).toFixed(2) + "%", false],
    ["Cycles run", nfmt(p.cycles), false],
  ];
  const dl = $("#stats-slot");
  dl.textContent = "";
  for (const [k, v, bad] of rows) {
    const box = el("div", "stat");
    const dd = el("dd", bad ? "down" : null);
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

function renderPositions(feed) {
  renderRows("#positions-slot", '[data-f="positions.count"]', feed.positions,
    "No open positions. The agent is flat.", (p) => {
      const r = el("div", "row");
      r.append(el("div", "row-q", p.question),
               el("div", "row-n", money(p.cost_basis)),
               el("div", "row-sub", `${p.side} · ${nfmt(p.qty)} @ ${cents(p.avg_cost)}`),
               el("div", "row-side", p.platform));
      return r;
    });
}

function renderTrack(feed) {
  renderRows("#track-slot", '[data-f="track.count"]', feed.track_record,
    "Nothing has settled yet. The track record starts at the first resolution.", (s) => {
      const r = el("div", "row");
      const n = el("div", "row-n" + (s.pnl < 0 ? " down" : ""), signed(s.pnl));
      r.append(el("div", "row-q", s.question), n,
               el("div", "row-sub", `resolved ${s.outcome} · ${s.ts.slice(0, 10)}`),
               el("div", "row-side", s.platform));
      return r;
    });
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
    const kind = /pangu/i.test(src) ? "our forecast" : "current temp";
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

function renderOpps(feed) {
  const rows = ((feed.board && feed.board.markets) || [])
    .filter((m) => m.loc && (m.state === "pending" || m.state === "held") && m.edge != null)
    .sort((a, b) => oppScore(b) - oppScore(a)).slice(0, 14);
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

function setLens(mode) {
  document.querySelectorAll(".lens").forEach((b) => b.classList.toggle("on", b.dataset.lens === mode));
  window.OTGlobe && OTGlobe.setLens(mode);
  const rail = $("#opprail"); if (rail) rail.hidden = mode !== "edge";
}

function globeTip(m, x, y) {
  const tip = $("#globe-tip");
  if (!tip) return;
  if (!m) { tip.hidden = true; return; }
  const g = GROUPS[m.place] || { items: [m], primary: m };
  const p = g.primary;
  tip.textContent = "";
  tip.append(el("b", null, m.place));
  const line = STATE_LABEL[p.state] + (p.mid != null ? ` · mkt ${cents(p.mid)}` : "") +
    (p.edge != null ? ` · edge ${cents(Math.abs(p.edge))}` : "");
  tip.append(el("span", "tip-detail", line), el("span", "tip-q", p.question));
  tip.append(el("span", "tip-more", g.items.length > 1
    ? `${g.items.length} markets here · click to see all` : "click for detail"));
  tip.hidden = false;
  const band = tip.parentElement.getBoundingClientRect();
  tip.style.left = Math.max(10, Math.min(x + 16, band.width - 300)) + "px";
  tip.style.top = Math.max(10, Math.min(y - tip.offsetHeight / 2, band.height - tip.offsetHeight - 12)) + "px";
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
  $("#strip") && $("#strip").classList.toggle("is-stale", stale);
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

  // The model gets a link when its weights are public — which, for an agent
  // arguing that the crowd is wrong, is the difference between a claim and a
  // checkable one.
  const model = $('[data-f="agent.forecaster"]');
  model.textContent = "";
  const { label, url } = feed.agent.forecaster;
  if (url) {
    const a = el("a", "strip-model", label);
    a.href = url;
    a.rel = "noopener";
    model.append(a);
  } else {
    model.textContent = label;
  }

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
    a.rel = "noopener";
    nav.append(a);
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
  renderHero(feed);
  renderCurve(feed);
  renderStats(feed);
  renderTheses(feed);
  renderPositions(feed);
  renderTrack(feed);
  renderRsi(feed);
  renderCompute(feed);
  renderLive(feed);
  renderGlobe(feed);
  renderOpps(feed);
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

  const feed = await fetchFeed();
  if (!feed) {
    $("#hero-slot").textContent = "";
    $("#hero-slot").append(el("p", "empty",
      "The feed did not load. The agent keeps trading either way — reload, or read feed.json directly."));
    return;
  }
  applyFeed(feed);

  setInterval(tickLive, 1000);                 // the "updated Ns ago" counter climbs live
  setInterval(async () => {                    // pull fresh data without a reload
    const f = await fetchFeed();
    if (f && f.generated_at !== SEEN.generated) applyFeed(f);
  }, 45000);
}

main();
