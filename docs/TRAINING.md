# Training a local forecaster on your journal

OpenThomas works with hosted APIs out of the box, but the journal it
accumulates is a fine-tuning dataset for a **local model that gets better on
the markets you actually trade**. This is the self-improvement path for users
with GPUs; nothing here phones home.

## Why bother

- Hosted frontier models are strong general forecasters but systematically
  miscalibrated per category — and you pay per token, forever.
- Research shows fine-tuning for calibration works (Halawi et al. 2024,
  arXiv:2402.18563), and OpenThomas already collects exactly the right data:
  `(question, rules, market price, your model's forecast, outcome)` for every
  settled market.
- A 12B-class open model (e.g. Gemma 12B) fine-tuned on a few thousand of your
  journal rows can match a much larger general model *on your market universe*,
  and runs on one consumer GPU. Larger open models (27B+, or bigger if your
  hardware allows) close more of the gap.

## Hardware guide

| Model class | Inference | LoRA fine-tune |
|---|---|---|
| ~12B (Gemma 12B) | 1× 24 GB GPU (Q4: 12 GB) | 1× 24 GB (QLoRA) |
| ~27–31B | 2× 24 GB or 1× 48 GB | 1–2× 48 GB (QLoRA) |
| 70B+ | 4× 24 GB+ | multi-GPU node |

## The pipeline

OpenThomas's self-improvement has two artifacts (docs/RSI.md): the **harness**
ships to GitHub, where a diff is reviewable; the **model** — weights, and the
data they were fit on — ships to Hugging Face, because a weight update is not
reviewable as text. The only way to judge one is to see its training set and
what it scored on days it never saw. So the pipeline pushes both, chained.

Point it at your org once:

```yaml
# ~/.openthomas/config.yaml
hub:
  org: openthomas        # "" (the default) disables every push
  dataset: ""            # defaults to <org>/journal
  private: false
```

```bash
pip install 'openthomas[hub]'        # publishing only; no torch
export HF_TOKEN=hf_...               # write-scoped
```

1. **Export the dataset** — settled markets only, first forecast per market,
   temporal split written into the rows:

```bash
openthomas dataset --out data/journal.jsonl        # local
openthomas dataset --push                          # and to <org>/journal
# ✓ Pushed to https://huggingface.co/datasets/openthomas/journal @ 4f1c9ab2e0d1
```

Keep that commit sha. It pins the exact rows, and step 3 refuses to publish
weights that cannot name it.

Each row: the prompt data block the forecaster actually saw, the market price
at the time, its probability and reasoning, and the label — `outcome`, `pnl`,
and `reward = pnl · exp(-0.05 · days_to_close)`. Third-party news headlines are
excluded (not ours to redistribute); `had_news` records that the model saw them.

2. **Fine-tune with QLoRA** (needs `pip install 'openthomas[train]'`):

```bash
python scripts/train/finetune_lora.py \
  --base google/gemma-3-12b-it \
  --data data/journal.jsonl \
  --out models/openthomas-lora
```

The objective is calibration: probability tokens whose implied Brier score on
the held-out split beats the base model. **Do not resplit the data.** The
`split` column is temporal and already assigned; a random split validates on
days the model trained on and will lie to you.

3. **Evaluate, then publish only if it won.**

```bash
python scripts/train/evaluate.py --model models/openthomas-lora --data data/journal.jsonl \
  --out eval.json     # {"brier_base": 0.251, "brier_tuned": 0.194, "n_validation": 128}

openthomas push-model --adapter models/openthomas-lora --name openthomas-lora-12b \
  --base google/gemma-3-12b-it --dataset-revision 4f1c9ab2e0d1 --eval eval.json
```

`push-model` refuses without held-out numbers and a dataset revision, and writes
them both into the model card — including when the adapter *lost*. A card
without evidence is a claim we cannot support.

4. **Serve it and point OpenThomas at it:**

```bash
vllm serve models/openthomas-lora --port 8000
openthomas init --provider openai --base-url http://localhost:8000/v1 --model openthomas-lora
```

Then set `site.model_label` and `site.model_url` so openthomas.com names what it
is actually thinking with, and links the weights.

## Ground rules (recursive self-improvement, with brakes)

- The forecaster is the only trainable component. **The risk engine is never
  learned** — sizing and caps stay deterministic no matter how good the model
  looks.
- Data volume matters: below ~500 settled forecasts, stick with Platt scaling
  (already automatic); fine-tuning on tiny samples memorizes noise.
- Never evaluate on markets the model saw resolve during training. The
  temporal split is assigned at export and carried in the `split` column, so
  the only way to leak is to deliberately ignore it.
- Re-run the evaluation after every market regime you care about (elections,
  Fed cycles); a fine-tune can rot.

Dataset export and publishing live in the package (`openthomas dataset`,
`openthomas push-model`). The LoRA and eval scripts land in `scripts/train/`
with the `[train]` extra as they stabilize — contributions welcome.
