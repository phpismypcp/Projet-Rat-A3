# PROGRESS — Anomaly Explainer

> A plain-language journal of **what** we are building, **why** each choice was
> made, and **where** we are. Written so you can explain the project to someone
> else without reading the code. Newest phase updates are appended at the bottom.

---

## 1. The problem in one paragraph

Banks process millions of card transactions. Almost all are legitimate; frauds
are extremely rare (in our dataset, **492 frauds out of 284,807 — about 0.17%**).
That rarity is exactly what makes classic "supervised" fraud models struggle:
there are too few fraud examples to learn from. On top of that, when a model
does flag a transaction, it usually gives only a **number** ("suspicion score =
0.87") with no reason an analyst can act on.

**Our system solves both halves:**
1. **Detect** anomalies without needing fraud examples to train on.
2. **Explain** each detection in plain language an analyst or auditor can trust.

---

## 2. The core idea (the "aha")

### Detection by reconstruction error

We train an **auto-encoder** — a neural network that learns to compress a
transaction into a small summary and then rebuild it — **using only normal
transactions**. Because it has only ever seen normal behaviour, it becomes very
good at rebuilding normal transactions and **bad at rebuilding fraudulent ones**
(it has never seen their patterns).

So the trick is:

> Feed any transaction in → let the model rebuild it → measure how wrong the
> rebuild is (the **reconstruction error**). **Big error = suspicious.**

We set a **threshold** on that error. Above the threshold → flagged as anomalous.
This is *unsupervised*: we never train on fraud labels; we only use the labels
at the very end to check how well we did.

### Why a *Transformer* auto-encoder (not a plain one)

The project spec requires a **Transformer** auto-encoder. A Transformer uses an
**attention mechanism**, which lets the model weigh how each feature relates to
every other feature (e.g. "a high amount *combined with* an unusual time *and* an
odd V14 value"). This captures complex interactions between variables better than
a simple network, giving cleaner reconstructions and sharper anomaly detection.

### Why an LLM for explanations

A reconstruction error is still just a number. So for every flagged transaction
we hand the details to a **Large Language Model** (running **locally** via
**Ollama**, so no transaction data ever leaves the machine — important for
banking privacy). The LLM turns the raw numbers into a human explanation:
*which* attributes drove the anomaly, *what* the suspicious behaviour looks like,
an estimated *risk level*, and *next steps* for a human analyst.

---

## 3. The dataset

- **Source:** Kaggle "Credit Card Fraud Detection" (`mlg-ulb/creditcardfraud`).
- **Size:** 284,807 transactions, 492 fraudulent (~0.17%).
- **Columns:**
  - `Time` — seconds elapsed since the first transaction.
  - `V1`…`V28` — 28 anonymised features (already transformed via PCA for privacy;
    we do not know what each one literally means).
  - `Amount` — the transaction amount.
  - `Class` — the label (0 = normal, 1 = fraud). **Used only for evaluation**,
    never for training the detector.

---

## 4. How the pieces fit (architecture)

```
   raw transaction
        │
        ▼
 ┌──────────────┐   scale Amount/Time, keep only NORMAL rows for training
 │ Preprocessing│
 └──────┬───────┘
        ▼
 ┌────────────────────────┐   attention encoder → small latent vector → decoder
 │ Transformer Auto-Encoder│   (trained ONLY on normal transactions)
 └──────┬─────────────────┘
        ▼
 ┌──────────────┐   reconstruction error vs statistical threshold → normal/anomaly
 │ Detection    │   + a severity score (how far above the threshold)
 └──────┬───────┘
        ▼
 ┌──────────────┐   builds a prompt from the errors, asks the local LLM (Ollama),
 │ LLM Explainer│   returns a readable explanation
 └──────┬───────┘
        ▼
 ┌──────────────┐   Streamlit dashboard: pick a transaction, see verdict,
 │ Analyst UI   │   per-feature deviations, and the explanation
 └──────────────┘
```

Each box is a separate, small module so it can be understood, tested, and
replaced independently.

---

## 5. Technology choices & the reasons

| Choice | Why |
|--------|-----|
| **Python + PyTorch** | Standard for building custom neural networks like the Transformer auto-encoder. |
| **scikit-learn** | Battle-tested scaling (`StandardScaler`) and metrics. |
| **Transformer auto-encoder** | Required by spec; attention captures feature interactions for better anomaly detection. |
| **Train on normal-only** | Sidesteps the extreme class imbalance — we never need fraud examples to learn. |
| **Statistical threshold** (`mean + k·σ`, or a high percentile) | Turns a continuous error into a clear yes/no decision, tunable to reduce false positives. |
| **PR-AUC / precision / recall** as metrics (not plain accuracy) | With 0.17% fraud, "99.8% accurate" is meaningless (predicting *always normal* scores that high). Precision/recall on the rare class is what matters. |
| **Ollama (local LLM)** | Explanations without sending sensitive banking data to a cloud API. `llama3.2:3b` is small enough to run on this CPU-only machine. |
| **Streamlit** | Fastest way to give analysts a visual, reusable interface for the demo/defense. |

---

## 6. Plan of phases

- **Phase 0 — Setup & data** *(in progress)*: project skeleton, dependencies,
  Ollama, dataset in place.
- **Phase 1 — Preprocessing**: validate schema, scale `Time`/`Amount`, split
  normal vs fraud, train/val/test split. Test-first.
- **Phase 2 — Transformer auto-encoder**: build + train on normal-only, save
  weights and the scaler.
- **Phase 3 — Detection engine**: reconstruction error (total + per-feature),
  threshold, severity score.
- **Phase 4 — Evaluation**: precision/recall/F1/PR-AUC; tune threshold to cut
  false positives.
- **Phase 5 — LLM explainer**: prompt → Ollama → structured explanation, with
  graceful fallback if the model is unavailable.
- **Phase 6 — Streamlit UI**: the analyst dashboard.
- **Phase 7 — Docs & polish**: README, diagram, final test coverage.

---

## 7. Progress log

### Phase 0 — Setup & data
- **Project skeleton created** — modular layout under `src/anomaly_explainer/`
  (`data`, `model`, `detection`, `evaluation`, `explain`) plus `app/`, `scripts/`,
  `tests/`. *Why: many small, focused modules are easier to understand, test, and
  change than a few big files.*
- **`config.py`** — every tunable value (paths, model size, training settings,
  threshold, LLM model name) lives in one place. *Why: no magic numbers scattered
  around; anyone can see and adjust the knobs in one file.*
- **Local git identity set to personal** (`Felix Drabble` / `felixdrabble@gmail.com`)
  so commits here don't use the work account. Global work identity left untouched.
- **Dependencies installed & verified** into a local virtual environment `.venv`
  (torch 2.13, pandas 3.0, scikit-learn 1.9, streamlit 1.59, pytest, matplotlib);
  all import cleanly.
- **Ollama installed** and responding; the `llama3.2:3b` model is being pulled
  (only needed from Phase 5 onward).
- **Dataset in place** — `data/creditcard.csv` present (150 MB, the full Kaggle
  set). A `scripts/download_data.py` helper also exists for re-fetching + schema
  validation.
- **Phase 0 complete.** Next: Phase 1 (preprocessing), written test-first.

### Phase 1 — Preprocessing *(complete)*
Written **test-first** (14 tests, 100% coverage on the data modules). Two small
modules under `src/anomaly_explainer/data/`:

- **`loader.py`** — reads the CSV and *validates the schema at the boundary*
  (all expected columns present, no nulls). Fails fast with a clear error rather
  than letting bad data flow downstream. *Why: never trust external data.*
- **`preprocess.py`** — three decisions that matter, and the reasoning:
  1. **Train on normal transactions only.** The auto-encoder must learn what
     "normal" looks like, so fraud rows are kept out of training and reserved
     for validation/test. *This is what makes the approach work despite only
     0.17% fraud.*
  2. **Fit the scaler on the training set only, then apply it everywhere.**
     Standardizing with statistics from the whole dataset would leak val/test
     information into training. We fit on train, transform the rest.
  3. **Immutable transforms** — functions return new frames/arrays and never
     modify their input.

- **Verified on the real 150 MB dataset:** 284,807 rows → 284,315 normal / 492
  fraud (0.173%). Split gives **199,020 normal-only training rows**, and val/test
  of ~42,893 rows each containing **246 frauds apiece** for threshold tuning and
  final evaluation. Training features standardize to mean≈0 / std≈1 as expected.

**Next:** Phase 2 — build and train the Transformer auto-encoder on the
normal-only training set.

### Phase 2 — Transformer auto-encoder *(code complete; training runs on a GPU machine)*
Written **test-first** (9 new tests; 23 total passing). Three small modules under
`src/anomaly_explainer/model/`:

- **`transformer_ae.py`** — the network. Each of the 30 features becomes a
  *token*; **multi-head self-attention** learns how features relate, so the model
  captures interactions a plain auto-encoder would miss. Flow: values →
  per-feature embedding → Transformer encoder → **compact latent vector** →
  Transformer decoder → reconstructed values. Poor reconstruction = anomaly.
- **`train.py`** — trains on the **normal-only** training set with MSE loss and
  **early stopping**, monitoring reconstruction loss on *unseen normal validation
  rows* so it halts when it stops generalizing. Auto-detects **CUDA (GPU)**,
  falling back to CPU.
- **`persistence.py`** — saves/loads three artifacts together (`model.pt`,
  `scaler.pkl`, `config.json`) so the exact architecture can always be rebuilt
  before loading weights.

**A practical note on where training runs.** On this CPU-only machine each epoch
took ~2 minutes; a full run is ~30–45 min. The loss *was* dropping correctly
(train 0.48 → 0.22, val 0.29 → 0.19 in two epochs), so the pipeline is verified —
but to keep iteration fast, **the actual training is done on a separate GPU
machine**. The code is GPU-aware and the README documents the hand-off: copy the
project + CSV over, run the train script, then copy `data/artifacts/` back here
for detection and the UI. All hyperparameters live in `config.py`.

**Next:** Phase 3 — detection engine (reconstruction error per feature, a
statistical threshold, and a severity score), which we can build and unit-test
now and point at the trained artifacts once they're back.

*(Later phases will be appended here as they are completed.)*
