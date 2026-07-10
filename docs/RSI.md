# Self-Improvement (RSI): OpenThomas updates OpenThomas

> The agent that *operates* OpenThomas (a human, or a coding assistant) must
> not be the thing that *improves* OpenThomas. Improvement is a loop the
> system runs on itself — on the operator's own compute — with the operator
> moved up the stack: setting bounds and reviewing lineage, not editing
> parameters by hand.

Framing follows Weng, [*Harness Engineering for Self-Improvement*](https://lilianweng.github.io/posts/2026-07-04-harness/)
(2026): near-term RSI is optimization of the **harness** — how the model
plans, acts, remembers, and is evaluated — and its first bottleneck is
evaluator strength. That is exactly why weather markets are the right
testbed: NOAA ground truth is objective, settlement is daily (dense
feedback), and the replay is reproducible and leak-free. OpenThomas has the
strong evaluator most RSI attempts lack.

## The structural guarantee: kernel / agent planes

Self-improvement is safe only if improvement cannot redefine "improvement".
So the codebase is split into two planes, and the split is the security
model:

| Plane | Contents | Who writes it |
|---|---|---|
| **kernel** | `openthomas/kernel/` (bounds, promotion gate), the risk engine code and all safety rails (sizing, exposure caps, kill-switch), the truth pipeline (`weather/verification.py`, `weather/replay.py::collect_rows`), the journal schema | The operator only. The evolution loop has **no write path** here. |
| **agent** | Decision-rule parameters and the forecast prompt template (today), workflow structure, strategy code, the evolver itself (roadmap) | The evolution loop, through the kernel gate. |

One line needs drawing precisely: `risk.min_edge` and `risk.market_prior_weight`
live on the RiskProfile config object but are entry-*selectivity* knobs — how
picky to be, not how much to risk. They are genome parameters under kernel
bounds. Sizing, exposure caps, and the kill-switch are safety rails and can
never enter `PARAM_SPACE`; a kernel policy test
(`test_param_space_never_touches_safety_rails`) fails any change that tries.

## Two artifacts, two homes

The planes above say *who may write what*. Cutting the same system the other
way — *what is produced* — gives two artifacts, and they are published in
different places because they are audited in different ways.

| Artifact | What it is | Where | How you check it |
|---|---|---|---|
| **harness** | gate, bounds, risk engine, decision rules, prompts, the evolver | [GitHub](https://github.com/PredictionMarketTrader/openthomas) | read the diff |
| **model** | LoRA adapters, and the journal rows they were fit on | [Hugging Face](https://huggingface.co/openthomas) | read the training set and the held-out score |

A code change is reviewable as text: you can look at a diff and say whether the
kill-switch still fires. A weight change is not. Nobody can read a LoRA and tell
you what it learned, so the only honest way to publish one is to ship the data it
was trained on and the number it scored on days it never saw. `openthomas
dataset --push` returns the dataset's commit sha; `openthomas push-model` refuses
to write a model card that does not cite it alongside a held-out Brier score.
The chain — weights → dataset revision → journal — is the audit.

Both artifacts face the same gate. A trained adapter is a candidate, not an
authority (docs/TRAINING.md).

The agent plane can change everything about *how*; it can never change *what
counts as good*. An optimizer that can edit its own judge will find it easier
to lower the bar than to clear it — so the judge is not on its edit surface.
The operator's ongoing role shrinks to kernel policy: widening or narrowing
`PARAM_SPACE` bounds, moving gate thresholds, adjusting risk caps, and
auditing the lineage.

Today the split is enforced by code structure (the LLM proposer only ever
returns JSON; every file write happens in deterministic code after the gate
has ruled). The deployment roadmap makes it structural in the OS: kernel
paths and `generations.json` owned by the operator's user, the evolution
process running as a user with write access only to the agent plane, gate
decisions crossing a process boundary. Every future promotion is then a git
commit authored by the loop itself — auditable by construction.

## Three nested loops

1. **Trading loop** (minutes, `agent/loop.py`): scan → forecast → risk-check
   → execute. Runs the *champion* parameters; never waits on the loops below.
2. **Reflection loop** (on settlements, `memory/lessons.py`): distills
   settled trades into playbook rules — content-level improvement, already
   bounded (ops, caps, per-rule track records).
3. **Evolution loop** (`improve/loop.py`): the harness improving the
   harness — the decision operator daily-ish, the forecast operator
   weekly-ish (its scoring pays a model call per sampled market):

```
mine failure evidence      journal stats, losing settlements
        │
propose mutations          LLM (local endpoint) directed by evidence
        │                  + Gaussian mutations drawn from the ARCHIVE
        │                    (not just the champion → diversity)
        ▼
kernel gate                leak-free replay, held-in picks / held-out vetoes,
        │                  min-trade-count floor, total-PnL scoring
        ▼
promote │ reject │ rollback
        │
generations.json           full lineage: parent, evidence, scores, status
improve-log.jsonl          every meta-cycle, auditable
```

This is the propose–evaluate–accept shape of Self-Harness, with a Darwin
Gödel Machine-style archive: rejected and rolled-back generations stay on
file as future mutation parents, against diversity collapse.

## The gate (kernel, frozen)

- **Held-in / held-out**: winners are picked on older days; the freshest
  settled days only get a veto. The window rolls forward daily, so repeated
  meta-cycles face genuinely new data instead of re-fitting one static split.
- **No peeking**: station bias/sigma are learned strictly before the replay
  window (`collect_rows`), prices come from snapshot candlesticks that
  precede the temperature extreme being bet on.
- **Anti-gaming by construction**: scoring is *total* replay PnL with a
  minimum trade count — a rule that trades twice can't win on ratio, and a
  rule that trades nothing can't win on drawdown.
- **Out-of-sample regret rollback**: each meta-cycle first re-scores the
  active generation against its parent on the fresh window. If the parent
  now wins, the promotion was overfit and is reverted before anything new is
  proposed.
- **Baseline never overridden silently**: generation 0 is the operator's
  config and applies no overrides; the human's settings win until the loop
  has promoted evidence-backed changes past them.

## Mutation operators: one gate, growing write surface

The three improvement axes (harness / weights / memory) are unified as
mutation operators feeding the *same* accept mechanism:

| Operator | Surface | Status |
|---|---|---|
| Playbook ops | lesson rules injected into prompts | shipped (`memory/lessons.py`) |
| Calibration refit | Platt parameters per category | shipped (`forecast/calibration.py`) |
| **Decision mutation** | `PARAM_SPACE` decision tags: min_edge, market-prior blend | **shipped (`improve/`)**, daily-ish |
| **Forecast mutation** | the forecast prompt template + anchor delta, scored by LLM-in-replay | **shipped (`improve/forecast_replay.py`)**, weekly-ish |
| Workflow-structure mutation | cycle structure as data, scanner filter config | next |
| Code diffs | agent-plane source, including the evolver itself | later: adds shadow paper-trading vs champion + operator review of the diff before merge |
| Weight updates | LoRA/GRPO on journal outcomes, time-discounted reward `R·exp(-λ·days_to_close)` | later: a trained adapter is just another candidate at the gate (`docs/TRAINING.md`) |

A parameter (or operator) is admitted only when the gate can actually
discriminate on it. Tuning what you cannot measure is drift, not
improvement — that rule decides what enters `PARAM_SPACE`, in which order
the operators ship, and why code diffs come last (they need the strongest
evaluation: replay *plus* shadow trading *plus* human review of the diff,
with the review moving up-stack over time, not disappearing).

### The forecast operator (LLM-in-replay)

The prompt template is a genome parameter like any other. A generation
stores concrete template text only when its operator actually evolved it;
an unevolved slot stays None — "track the built-in default" — so upstream
fixes to the stock template keep reaching deployments instead of being
frozen by whichever generation happened to be promoted last. Structure is
kernel policy (`TextSpec`: required placeholders whitelisted by a format
parse — which also closes the `{question.__class__}`-style str.format
injection channel — the `"probability"` parser contract, length cap);
content is evolved. Scoring rebuilds each settled
market's prompt from as-of archived data (guidance consensus, station stats,
snapshot quotes — never the outcome), gets the model's probability at
temperature 0, applies the live anchor clip, and pushes it through the
champion's decision rule frozen — so the gate's verdict attributes wins to
the mutation and nothing else. Promotion additionally requires calibration
not to degrade (`beats_forecast`: Brier veto) — a prompt that wins PnL while
worsening Brier got lucky, and luck does not promote.

A cold forecast meta-cycle is ~500 local-model calls — hours of wall clock,
not seconds. It therefore runs on a background worker: the trading loop only
ever starts it and walks away, because a fast loop that waits on the slow
loop leaves open positions unmarked and the drawdown kill-switch unevaluated
for a dozen cycles. One meta-cycle runs at a time, it opens its own journal
(a sqlite connection belongs to the thread that opened it), and promotion
mutates the live `Settings` in place, so the next trading cycle picks it up.

Every candidate faces the same kernel-sampled rows, and model outputs are
cached by (template, model, market): anchor-delta-only mutations re-score
from cache with zero model calls, and repeated meta-cycles only pay for
newly settled days. Mutations are restricted to the invoking operator's
tags — a decision cycle can never smuggle in an unscored prompt change, and
vice versa; adjacent generations therefore differ only in one operator's
parameters, which is what makes the rollback check a controlled comparison.

Known fidelity limits, accepted rather than hidden: replay has no news
brief, no intraday observations, no NWS discussion, and no calibration
layer — it measures the template's skill at adjusting the statistical
baseline, the dominant live pathway for weather markets. (The journal now
archives each live forecast's data/news inputs, so a future trace-replay
evaluator can close this gap for generic markets.) Residual leak risk: a
model could in principle "remember" recent weather from training data — with
a local model whose cutoff precedes the replay window this is negligible,
but swap the forecaster and the assumption must be rechecked.

## Failure modes and their counters

- **Reward hacking** → the judge (gate, truth pipeline, risk engine) is off
  the edit surface; hard bounds; totals-not-ratios scoring.
- **Overfitting the evaluator** → rolling held-out veto + regret rollback on
  post-promotion data.
- **Diversity collapse** → archive-parent random mutations alongside
  LLM-directed ones.
- **Weak evaluator** → domain choice: daily-settling weather markets with
  NOAA truth; operators without replay data get "insufficient replay data",
  not silent promotions.
- **Runaway autonomy** → drawdown kill-switch, paper-by-default, and live
  mode's two-step opt-in are all kernel; the evolution loop cannot loosen
  them. A dead LLM endpoint degrades evolution to random mutations; a dead
  evolution loop degrades OpenThomas to a static (still disciplined) trader.
