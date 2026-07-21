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
- **Phase 4 — Evaluation** *(complete)*: precision/recall/F1/PR-AUC; tune
  threshold to cut false positives.
- **Phase 5 — LLM explainer** *(complete)*: prompt → Ollama → structured explanation, with
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

### Phase 3 — Detection engine *(complete)*
Written **test-first** (22 new tests; 45 total passing, **99% coverage**, the
detection modules at 100%). Three small modules under
`src/anomaly_explainer/detection/`:

- **`reconstruction.py`** — turns the model's rebuild quality into numbers, at
  two granularities: **per-feature error** (which attributes were badly rebuilt —
  fuel for the LLM explanation) and **total error** (the mean, used to decide).
  Runs batched so it handles the full 284k-row dataset without memory blow-ups,
  and is GPU-aware.
- **`threshold.py`** — the rule that turns a continuous error into a yes/no call.
  Two strategies: **`sigma`** (`mean + k·std` of normal errors) and
  **`percentile`** (top *X%* most poorly reconstructed). The threshold is fit on
  *normal* errors only, and tuning it is the main lever for trading false alarms
  against missed frauds.
- **`engine.py`** — the `AnomalyDetector`: for each transaction it produces the
  error, the anomaly flag (`error > threshold`), and a **severity score**
  (`error / threshold`, so `> 1` means anomalous). `top_contributors()` lists the
  features most responsible for an anomaly — the bridge to the explainer.

Everything here is deterministic and unit-tested against a fresh model on
synthetic data, so it already works; once the trained `data/artifacts/` come back
from the GPU machine, the detector plugs straight in with real weights.

**Next:** Phase 4 — evaluation (precision/recall/F1/PR-AUC on the labelled
val/test sets, and a threshold sweep to minimise false positives), which is where
we finally use the real trained model.

### Phase 4 — Evaluation & threshold tuning *(complete)*
Written **test-first** (29 new tests; 74 total passing, **99% coverage**, the
evaluation modules at 100%). This is the phase where the real trained model was
finally measured — and it exposed a serious bug in our own defaults.

#### The headline finding: our default threshold was catching almost nothing

The model itself was fine. The *rule we used to read it* was broken.

Reconstruction errors on normal transactions are **extremely right-skewed**:
mean `0.0442`, but median only `0.0074`, with std `3.19`. A handful of hard-to-
rebuild normal transactions drag the mean and the standard deviation far above
where the bulk of the data actually sits. The textbook `mean + 3·σ` rule
therefore lands at **9.60** — deep in the tail, above almost every fraud too.

> With the old `sigma` default the detector caught **15 of 246 frauds (6% recall,
> F1 = 0.114)**. The model was working; the threshold was throwing its answers away.

We also tested a robust variant (`median + k·MAD`) in case the skew was the only
problem — it peaked at F1 ≈ 0.67, still worse than percentiles. So the fix is to
threshold by **percentile of the normal error distribution**, which is immune to
the skew by construction: it asks "what do the worst 0.1% of normal transactions
look like?" rather than assuming a bell curve that isn't there.

#### Choosing the operating point on evidence

`evaluation/sweep.py` walks candidate percentiles on the **validation** split and
reports the trade-off at each, so the choice is made from a table, not a guess:

| percentile | threshold | TP | FP | precision | recall | F1 |
|-----------:|----------:|---:|---:|----------:|-------:|----:|
| 99.0 | 0.322 | 199 | 427 | 0.318 | 0.809 | 0.456 |
| 99.5 | 0.492 | 198 | 214 | 0.481 | 0.805 | 0.602 |
| 99.8 | 0.863 | 188 |  86 | 0.686 | 0.764 | 0.723 |
| **99.9** | **1.263** | **170** | **43** | **0.798** | **0.691** | **0.741** |
| 99.95 | 1.804 | 130 | 22 | 0.855 | 0.528 | 0.653 |

The table makes the trade-off concrete: loosening to the 99th percentile buys
12 extra frauds but costs **~10x more false alarms**. `99.9` is the F1-optimal
balance and is now the configured default.

#### Final result on the held-out test set

The threshold was chosen on validation only, then the test split — untouched
until this point — was scored **once**. Its numbers are therefore an unbiased
estimate rather than a number we tuned toward:

| Metric | Test set (42,894 rows, 246 frauds) |
|---|---|
| Precision | **0.842** |
| Recall | **0.715** |
| F1 | **0.774** |
| PR-AUC | **0.742** (random baseline: 0.0057 — a **129x** lift) |
| ROC-AUC | 0.964 |
| False positives | **33 out of 42,648 normal transactions (0.08%)** |

Two things worth noting. First, test F1 (0.774) came out slightly *above*
validation F1 (0.741), which is the sign we wanted: the threshold was not overfit
to the validation split. Second, that 0.08% false-positive rate is the project
spec's *"réduction drastique du nombre de faux positifs"*, delivered — an analyst
reviewing this queue sees roughly 4 alerts to find 3 real frauds.

Against the old `sigma` default on the same test set, F1 went from **0.114 to
0.774** — a **6.8x** improvement from changing one configuration decision.

#### What was built

- **`evaluation/metrics.py`** — confusion counts plus precision / recall / F1 /
  PR-AUC / ROC-AUC. Rates are computed as properties from the counts rather than
  stored, so they cannot drift out of sync. Validates labels and scores at the
  boundary (length, emptiness, binary labels).
- **`evaluation/sweep.py`** — the percentile sweep, with two selectors:
  `best_by_f1()` (balanced) and `best_at_min_precision(p)` (maximise recall
  subject to an analyst-workload budget).
- **`detection/threshold.py`** — now records *how* a threshold was derived
  (`parameter` = k or percentile) and can **save/load `threshold.json`**. This
  matters: the UI and explainer must reuse the exact tuned value, never silently
  re-derive their own.
- **`scripts/evaluate.py`** — runs the whole procedure, persists the threshold,
  and writes `evaluation_report.json` + `pr_curve.png` for the defense.
- **`config.py`** — default method switched to `percentile` at `99.9`, with the
  reasoning recorded in the docstring so the choice isn't a mystery later.

**Next:** Phase 5 — the LLM explainer. Two prerequisites identified while
reviewing the spec against the code:
1. `reconstruction.py` currently returns only *errors*, but the spec requires the
   LLM to also see the **reconstructed values**; a function returning them needs
   adding.
2. Values must be **inverse-transformed** back to real units before reaching the
   LLM, so an explanation reads "Amount = €4,821" instead of a z-score.

#### Ollama — resolved and measured

The Phase 0 model pull had never actually completed (`/api/tags` returned an
empty list), which would have blocked Phase 5. Now fixed and verified end to end:

- **`llama3.2:3b` pulled** (2.0 GB, Q4_K_M quantisation, 3.2B parameters). The
  Ollama service is `active` and `enabled`, so it survives reboots.
- **Verified it does the actual job**, not just that it responds: given a
  realistic payload (severity, per-attribute errors, original vs reconstructed
  values) it returned valid JSON containing exactly the four fields the spec
  requires — `attributes`, `interpretation`, `risk_level`, `suggestions` — and
  correctly rated the example transaction `high` risk.
- **Speed on this CPU-only machine:** ~10 tokens/s → **6-9 s per warm
  explanation**, ~19 s cold. `size_vram: 0` confirms pure CPU inference.

Two settings were added to `ExplainerConfig` as a direct result of measuring:

1. **`keep_alive = "30m"`.** Ollama unloads an idle model after 5 minutes by
   default. During a live defense, any pause longer than that would silently
   make the next explanation take ~19 s instead of ~7 s. Verified the override
   works (model expiry moved from 5 to 30 minutes).
2. **`json_format = True`** — Ollama constrains decoding to valid JSON, so the
   explanation can be *parsed* rather than scraped out of prose with a regex.

Storage note: the systemd service keeps models on the **root** partition
(`/usr/share/ollama/.ollama/models`), now at 73% used with 7.9 GB free, while the
264 GB of free space sits on `/home`. Fine for this 2 GB model; worth knowing
before trying a larger one.

The explainer will still ship with a template fallback so the tests and the UI
degrade gracefully if Ollama is ever unavailable.

### Phase 5 — LLM explainer *(complete)*
Written **test-first** (54 new tests; 128 total passing, **99% coverage**, all
`explain/` modules at 100%). This is the half of the project that turns a score
into something an analyst can act on.

#### Two gaps found by re-reading the spec against the code

Before writing the explainer we checked what §4.2 actually demands the LLM be
shown, and the detection engine could not yet supply all of it:

1. **The reconstruction itself was never exposed.** `reconstruction.py` returned
   only *errors*. But "the model expected an amount near 88 and saw 1809" is far
   more useful to an analyst than "the error on Amount was 12.1". Added
   `reconstruct()`, and `per_feature_errors()` is now derived from it, so the
   errors and the rebuilt values can never disagree.
2. **Values were in z-scores.** The detector works on standardized features, so
   a raw value reaching the LLM would read `Amount = 4.7`. Everything is now
   **inverse-transformed back to real units** before it reaches the prompt.

#### What was built (four small modules under `explain/`)

- **`context.py`** — assembles the evidence bundle for one transaction: the four
  inputs the spec names (original values, reconstruction, per-attribute error,
  severity), in real units, with attributes ranked by error.
- **`prompt.py`** — renders the prompt. Two deliberate choices: it presents
  *evidence rather than conclusions* (it never says "this is fraud", or the
  explanation would just parrot our own verdict), and it **states that V1–V28
  are anonymised PCA components** so the model doesn't invent confident stories
  about what `V14` means. Output language is configurable, French by default.
- **`ollama_client.py`** — the HTTP client. Its job is to fail *clearly*: a dead
  server, a missing model and a slow generation are three different problems
  with three different fixes, so they get three different messages.
  `is_available()` checks the server **and** that the model is installed — the
  precise failure we hit in Phase 0, where the server was running happily with
  an empty model list.
- **`explainer.py`** — orchestration, built for a 3B model on CPU that will
  sometimes misbehave. It tolerates a bare string where a list was requested,
  re-derives the risk level from measured severity if the model invents one, and
  falls back to a deterministic template on *any* failure. Every result carries
  `source` (`"llm"` or `"fallback"`) so the UI can be honest about where the
  words came from.

#### Verified end to end, not just unit-tested

`scripts/explain_transaction.py` runs the whole pipeline against the real
trained model, the tuned threshold, and live Ollama. On the most severe true
fraud in the test set (row 42823, error 67.94 vs threshold 1.26, **severity
53.8**) it produced a coherent French explanation, correctly rated it `high`
risk, and — notably — respected the PCA caveat, describing "composantes PCA
anonymisées" instead of fabricating a meaning for `V17`.

The `--no-llm` and `--false-positive` paths were exercised too. The worst false
positive (row 12626, severity 31.0) is a genuinely odd-looking normal
transaction, which is a useful thing to be able to show during the defense.

**A note on test discipline.** One test demanded the fallback quote a real
monetary value and initially failed: the fallback described only the top
contributor, which was `V17` — a number meaningless to a human. Rather than
relax the test, we fixed the behaviour: the fallback now also surfaces `Amount`
or `Time` whenever they are among the offending attributes, because those are
the only features an analyst can actually reason about.

**Next:** Phase 6 — the Streamlit dashboard. Design constraint already measured:
an explanation costs 6–9 s on CPU, so the UI must cache per transaction and show
a spinner rather than blocking, and detection over the full test set should be
computed once and cached with `st.cache_resource` / `st.cache_data`.

### Interlude — Does attention actually help? A baseline *(complete)*

The spec mandates a Transformer auto-encoder because attention should capture
interactions between features. That is a **claim**, and until now we had no
control for it: "PR-AUC 0.742" had nothing to be better *than*. This is the
single question a jury is most likely to ask, so we answered it with a
measurement instead of an assertion.

**How the comparison was made fair.** Everything is held constant except the
architecture: the same splits, the same seed, the same training loop (optimizer,
batching, early stopping), the same latent size (16), and the same thresholding
rule (99.9th percentile of normal validation errors). `train_autoencoder()` was
refactored to accept a `model_factory` precisely so the baseline reuses the
*identical* loop rather than a re-implementation of it.

Two baselines, because "the Transformer won" is weak if it merely had more
capacity:

| Model | Params | Epochs | PR-AUC | Precision | Recall | F1 | False positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Transformer AE (spec)** | 269,137 | — | **0.7422** | 0.842 | 0.715 | **0.774** | **33** |
| Dense AE (matched bottleneck) | 9,198 | 67 | 0.6635 | 0.722 | 0.549 | 0.624 | 52 |
| Dense AE (matched capacity) | 302,638 | 13 | 0.7138 | 0.722 | 0.654 | 0.687 | 62 |

**The Transformer wins, and not merely by being bigger** — it beats a dense model
with *more* parameters (302k vs 269k). Margin over the best baseline: **+0.0284
PR-AUC**.

**The most defensible number is the false-positive count: 33 vs 62.** At an
identical thresholding rule the Transformer raises roughly **half the false
alarms** while also catching more fraud. That is the spec's *"réduction drastique
du nombre de faux positifs"*, now demonstrated against a control rather than
asserted.

#### The most interesting finding: better reconstruction ≠ better detection

The wide dense auto-encoder reached a **much lower reconstruction loss** than the
narrow one (validation 0.031 vs 0.126) and yet **detected worse** (more false
positives, lower F1). This is the classic over-capacity failure mode: given
enough width, an auto-encoder learns to rebuild *everything* accurately —
including the frauds it was never trained on — which collapses the very error gap
detection depends on.

The lesson is that the bottleneck is not an inconvenience to be minimised; it is
the mechanism. Optimising reconstruction loss and optimising anomaly detection
are different objectives, and past a point they actively conflict.

#### A methodology bug this exercise caught

The first run capped baselines at 30 epochs. The small dense model hit that cap
while still improving (val loss falling monotonically, 0.135 → 0.126), so it was
*cut off rather than converged* and scored an unfairly low 0.609 PR-AUC. Given
proper time it converges at 67 epochs and reaches 0.6635.

The conclusion did not change, but the near-miss is the point: a baseline that is
quietly under-trained makes any comparison flattering to the proposed method.
`scripts/compare_baseline.py` now takes `--epochs` and records a `converged` flag
per model, so this cannot silently recur.

#### Honest limits of this result

- **One seed, one split.** With only 246 frauds in the test set, +0.028 PR-AUC is
  a modest margin and no confidence interval was computed. The false-positive
  gap (33 vs 52-62) is the more robust difference.
- The Transformer was trained on the separate GPU machine; the baselines here on
  CPU. Same loop, config and seed, but not the same hardware.

*(Later phases will be appended here as they are completed.)*
