"""Hugging Face: the model plane of RSI.

OpenThomas improves itself along two artifacts. The **harness** — gate, risk
engine, decision rules, prompts — is code, and it ships to GitHub where a diff
is reviewable. The **model** — weights, and the data they were fit on — ships
here, because a weight update is not reviewable as text: the only way to judge
it is to see what it was trained on and what it scored.

So the two are pushed as a chain. `push_dataset` returns the commit it created;
`push_adapter` refuses to publish weights that do not name that commit and the
held-out numbers that justify them. A model card without evidence is a claim
we cannot support, and this pipeline will not write one.

Requires `pip install 'openthomas[train]'` and a write-scoped `HF_TOKEN`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import Settings
from .dataset import SCHEMA_VERSION, summary, trainable, write_jsonl

# Evidence a model card must carry before any weights go public.
REQUIRED_EVAL = ("brier_base", "brier_tuned", "n_validation")


class HubError(RuntimeError):
    """A push could not proceed: no token, no org, or no evidence."""


def _api(token: str | None = None):
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - exercised by the extra, not tests
        raise HubError(
            "Hugging Face publishing needs the optional dependency: "
            "pip install 'openthomas[hub]'"
        ) from None
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise HubError("set HF_TOKEN to a write-scoped Hugging Face token")
    return HfApi(token=token)


def dataset_repo(settings: Settings) -> str:
    if not settings.hub.org:
        raise HubError("set hub.org in ~/.openthomas/config.yaml to publish")
    return settings.hub.dataset or f"{settings.hub.org}/journal"


def aliases(settings: Settings) -> dict[str, str]:
    """The one place that knows the model's public name. The journal records the
    endpoint's id; `site.model_label` is what a reader should see. Sharing this
    with the website keeps a serving alias from escaping through the dataset."""
    if settings.site.model_label:
        return {settings.forecaster.model: settings.site.model_label}
    return {}


# --- cards ---------------------------------------------------------------------

def dataset_card(stats: dict, settings: Settings) -> str:
    span = " → ".join(stats["span"]) if stats["span"] else "no settlements yet"
    thin = stats["trainable_rows"] < 500
    return f"""---
license: mit
task_categories:
- tabular-regression
tags:
- prediction-markets
- forecasting
- calibration
- weather
- build-in-public
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
---

# OpenThomas journal

Every settled market [OpenThomas](https://openthomas.com) has traded: the first
forecast it made, the price the market was offering when it made it, and how the
world resolved. Regenerated from the live journal and pushed as the agent trades,
so each commit is a timestamped record of what was known when.

The harness that produced this is at
[{settings.site.github.split("github.com/")[-1]}]({settings.site.github}).

| | |
|---|---|
| rows | {stats["rows"]} |
| trainable rows (prompt reconstructible) | {stats["trainable_rows"]} |
| rows with a market price at forecast time | {stats["rows_with_market_price"]} |
| train / validation | {stats["train"]} / {stats["validation"]} |
| span | {span} |
| YES base rate | {stats["base_rate"]} |
| forecaster | {", ".join(stats["forecasters"]) or "—"} |
| schema version | {SCHEMA_VERSION} |

## What is in a row

`question`, `category`, `data` (the statistical baseline and model guidance the
forecaster was shown), `p_market` (price at forecast time), `p_forecast`,
`reasoning`, `why`, `invalidation` — and the label: `outcome`, `pnl`, and
`reward` = `pnl · exp(-0.05 · days_to_close)`.

## Three rails, baked in

**Settled markets only.** An open position has no label, and publishing a live
view of one would hand away the trade. A settled market has no alpha left.

**First forecast per market.** Later forecasts on the same market watched the
price move and the day advance. Training on them teaches hindsight.

**The split is temporal and it travels with the data.** `validation` is the most
recent slice by time, assigned at export. A random split lets a model validate on
days it trained on, and it will lie to you about its Brier score.

## What is missing, on purpose

`news` is absent: the live prompt carried third-party headlines this repository
has no right to redistribute. `had_news` records that the forecaster saw them, so
nobody mistakes these rows for a faithful prompt replay. Rows with an empty `data`
field, or a null `p_market`, predate the journal archiving those inputs — they
carry a label and no features, and are excluded from `trainable rows` above.
{'''
## Not yet enough

Fewer than 500 trainable rows. Fine-tuning on a sample this small memorizes
noise; the agent uses Platt scaling until the journal is deeper. This dataset is
published as it fills, not because it is ready.
''' if thin else ''}
*Paper trading. Prediction market trading can lose all the money you allocate;
none of this is financial advice.*
"""


def model_card(name: str, base_model: str, ds_repo: str, ds_revision: str,
               evaluation: dict, settings: Settings) -> str:
    delta = evaluation["brier_base"] - evaluation["brier_tuned"]
    verdict = "beats" if delta > 0 else "does not beat"
    return f"""---
license: mit
base_model: {base_model}
datasets:
- {ds_repo}
tags:
- prediction-markets
- forecasting
- calibration
- lora
---

# {name}

A LoRA adapter for `{base_model}`, fit on [{ds_repo}](https://huggingface.co/datasets/{ds_repo})
at revision [`{ds_revision[:12]}`](https://huggingface.co/datasets/{ds_repo}/tree/{ds_revision}).
Trained by [OpenThomas](https://openthomas.com), which trades weather markets on
Kalshi and Polymarket and publishes every claim before it settles.

## Held-out result

The objective is calibration, so the number that matters is Brier score on the
temporal validation split — days the adapter never trained on.

| | Brier |
|---|---|
| base `{base_model}` | {evaluation["brier_base"]:.4f} |
| this adapter | {evaluation["brier_tuned"]:.4f} |
| improvement | {delta:+.4f} |

Measured on {evaluation["n_validation"]} held-out settlements. The adapter
**{verdict}** the base model out-of-sample.

Reproduce it: the dataset revision above pins the exact rows, and the split is
carried in the data, so there is nothing to reshuffle.

## What this cannot do

It forecasts. It does not size, and it cannot. In OpenThomas the risk engine —
fractional Kelly, exposure caps, the drawdown kill-switch — is deterministic and
is never learned, no matter how good a model looks. An adapter is a candidate at
a promotion gate, not an authority ([docs/RSI.md]({settings.site.github}/blob/main/docs/RSI.md)).

*Prediction market trading can lose all the money you allocate; none of this is
financial advice.*
"""


# --- pushes --------------------------------------------------------------------

def push_dataset(settings: Settings, rows: list[dict], token: str | None = None,
                 repo_id: str | None = None) -> str:
    """Upload train/validation splits plus a card. Returns the commit sha —
    the handle a model card must cite to claim it trained on this data."""
    from huggingface_hub import CommitOperationAdd

    api = _api(token)
    repo_id = repo_id or dataset_repo(settings)
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                    private=settings.hub.private, exist_ok=True)

    stats = summary(rows)
    scratch = Path(settings.home) / "hub"
    train = write_jsonl([r for r in rows if r["split"] == "train"], scratch / "train.jsonl")
    valid = write_jsonl([r for r in rows if r["split"] == "validation"],
                        scratch / "validation.jsonl")
    card = scratch / "README.md"
    card.write_text(dataset_card(stats, settings))

    info = api.create_commit(
        repo_id=repo_id, repo_type="dataset",
        operations=[
            CommitOperationAdd("README.md", str(card)),
            CommitOperationAdd("data/train.jsonl", str(train)),
            CommitOperationAdd("data/validation.jsonl", str(valid)),
        ],
        commit_message=f"journal @ {stats['rows']} settled markets "
                       f"({stats['trainable_rows']} trainable)",
    )
    return info.oid


def push_adapter(settings: Settings, name: str, adapter_dir: str | Path, base_model: str,
                 dataset_revision: str, evaluation: dict, token: str | None = None) -> str:
    """Publish trained weights — but only alongside what they scored.

    The org card promises this pipeline does not make claims it cannot support.
    Refusing here is how that promise is kept rather than merely written down.
    """
    missing = [k for k in REQUIRED_EVAL if k not in evaluation]
    if missing:
        raise HubError(
            f"refusing to publish weights without evidence: missing {', '.join(missing)}. "
            "Evaluate on the temporal validation split first."
        )
    if not dataset_revision:
        raise HubError("refusing to publish weights that do not name the dataset "
                       "revision they were fit on")
    adapter = Path(adapter_dir)
    if not adapter.is_dir():
        raise HubError(f"no adapter directory at {adapter}")

    api = _api(token)
    repo_id = f"{settings.hub.org}/{name}" if "/" not in name else name
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=settings.hub.private, exist_ok=True)

    ds_repo = dataset_repo(settings)
    (adapter / "README.md").write_text(
        model_card(name.split("/")[-1], base_model, ds_repo, dataset_revision,
                   evaluation, settings))
    (adapter / "openthomas_eval.json").write_text(json.dumps(evaluation, indent=1) + "\n")

    api.upload_folder(
        folder_path=str(adapter), repo_id=repo_id, repo_type="model",
        commit_message=f"{name}: Brier {evaluation['brier_base']:.4f} → "
                       f"{evaluation['brier_tuned']:.4f} on "
                       f"{evaluation['n_validation']} held-out settlements",
    )
    return f"https://huggingface.co/{repo_id}"


def pull_dataset(repo_id: str, revision: str = "main", token: str | None = None):
    """Fetch a pinned revision for training. Pinning is the point: a training
    run that cannot name its data cannot be reproduced or audited."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                             token=token or os.environ.get("HF_TOKEN"))


__all__ = ["HubError", "dataset_card", "dataset_repo", "model_card", "pull_dataset",
           "push_adapter", "push_dataset", "trainable"]
