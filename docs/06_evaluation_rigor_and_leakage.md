# 6. Evaluation, rigor & the leakage story

This is the most important chapter. The headline of EchoHawk is not a number —
it is **how the numbers were made honest**. If you read only one technical
chapter, read this one.

## 6.1 How we measure a detector

- **Train/test split.** We fit the model on one part of the data and measure it
  on a *separate* part it has never seen. Measuring on data the model trained on
  tells you nothing.
- **ROC-AUC.** Sweep the decision threshold and plot detection rate vs
  false-alarm rate; the area under that curve (0.5 = chance, 1.0 = perfect) is a
  threshold-independent score.
- **Pd at a fixed Pfa.** Because operators can't tolerate many false alarms, we
  report the **probability of detection at 1%, 5%, and 10% false-alarm rates** —
  the metric that actually matters for a security system.
- **Confusion matrix.** A table of right/wrong calls per class.

These live in `evaluate.py` (`plot_roc`, `pd_at_pfa`, `plot_confusion`) and are
used identically for synthetic and real data.

## 6.2 The trap: data leakage

**Data leakage** is when information from the test set sneaks into training,
making a model look better than it really is. It is the single most common reason
published machine-learning results fail to reproduce in the real world.

EchoHawk hit a textbook example, and the way it was caught and fixed is the
project's signature contribution.

### What happened

The real **DroneAudioDataset** stores drone audio as many **1-second files** —
but those short files were **pre-cut from a smaller number of long, continuous
recording sessions**. The drone class has **1,332 files** that come from only
**257 actual flight sessions** (their filenames share a prefix, e.g.
`B_S2_D1_067-bebop_000_.wav`, `_001_`, `_002_`…).

A naïve split shuffles the *clips*. So clip `_000_` of a session could land in
**training** while clip `_002_` of the **same continuous recording** lands in
**testing**. The two are nearly identical — same drone, same background, same
moment — so the model can "recognise the session" rather than learn the drone
signature. The test score is then **inflated**: the model is effectively peeking.

### How it was caught

Two clues. First, the numbers looked *too* good (CNN AUC 0.997, Pd@1%FA 0.95).
Second, when we switched from a clip-level split to a **file-level** group split,
the scores **barely moved** — which was suspicious, because each file is already
~1 second, so grouping by file is almost the same as grouping by clip. Inspecting
the filenames revealed the real unit of leakage: the **session**.

### The fix

We group by **recording session** (strip the chunk index from the filename) and
use a **`GroupShuffleSplit`** so that **every clip from one session stays
entirely in train or entirely in test** — never both. Negatives (each an
independent file) are grouped per file. Now the model cannot peek.

## 6.3 The honest numbers

Sealing the leak lowered the scores to trustworthy values — and that drop *is*
the result:

| Metric | RF (file-grouped, leaky) | RF (session-grouped, honest) | CNN (file-grouped, leaky) | CNN (session-grouped, honest) |
|---|---|---|---|---|
| ROC-AUC | 0.9872 | **0.9815** | 0.9968 | **0.9941** |
| Pd @ 1% FA | 0.7964 | **0.7453** | 0.9519 | **0.9384** |

The Random Forest's detection rate at 1% false alarm fell by ~5 points once it
could no longer recognise the session background. The CNN held up better,
suggesting it generalises more genuinely — but both numbers are now *earned*.

## 6.4 Honesty practices used throughout

- **Two benchmarks, clearly labelled.** Synthetic with *hard* confusers
  (vehicles) and real with *easy* negatives — never blended, never confused.
- **Report the operating point, not just AUC** (Pd@1/5/10% FA).
- **Fixed random seeds** so anyone can reproduce the exact split.
- **Show the leaky numbers too**, so the size of the inflation is visible.
- **State limitations plainly** (easy negatives; mostly-synthetic DOA;
  single-source, far-field).

## 6.5 Why this matters for a real system

A counter-drone system that was secretly "recognising the test site" would fail
the moment it met a new environment. Catching this class of error *before*
deployment is exactly the discipline a real program needs — and it's the most
transferable lesson in the whole project.

➡ Next: [Getting started](07_getting_started.md).
