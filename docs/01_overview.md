# 1. Overview — what EchoHawk is and why it exists

## The problem

Small drones are cheap, quiet, and hard to see. They are used for good things
(photography, inspection, delivery) but also for dangerous ones (smuggling,
spying, attacks). Protecting an area — an airport, a border, an event, a
sensitive site — increasingly means being able to **detect a drone before it
arrives** and know **where it is**. This field is called **counter-UAS**
(counter–Unmanned Aerial Systems), or "counter-drone".

There are several ways to spot a drone:

- **Radar** bounces radio waves off it — but small drones reflect very little,
  and radar can struggle near the ground and among clutter.
- **Cameras** can see it — but only in good light, with a clear line of sight.
- **Radio (RF)** can pick up the drone's control link — but some drones fly
  autonomously and emit almost no radio.
- **Sound (acoustics)** — the approach EchoHawk takes. A drone's propellers are
  *always* making noise, the system emits nothing itself (it is **passive** and
  therefore hard to notice), and it is cheap. Its weaknesses are shorter range
  and sensitivity to wind and background noise — so in practice acoustics
  **complements** the other methods.

EchoHawk is a clean, well-documented reference implementation of the acoustic
approach: a demonstrator of the core algorithms, not a finished product.

## What EchoHawk actually does

It is a pipeline with four stages:

1. **Detect** — given a short sound clip, decide *drone* or *not a drone*.
2. **Identify the signature** — estimate the drone's **blade-passing frequency**
   (a fingerprint of the propellers).
3. **Locate** — using several microphones (an *array*), estimate the
   **direction** the sound is arriving from.
4. **Track** — follow that direction smoothly as the drone moves.

## What's in the box

- A **library** (`echohawk/`) implementing every algorithm.
- Seven **runnable demos** (`examples/`) that reproduce all the figures.
- A built-in **simulator**, so the whole thing runs with **no downloads**.
- Hooks to evaluate on **real public datasets** (recorded drone audio; a
  microphone-array dataset called DREGON).
- **Tests**, continuous-integration, a Docker file, and a paper-style
  [technical note](../report/technical_note.md).

## Headline results (plain language)

- On realistic synthetic data where drones must be told apart from **ground
  vehicles** (which also hum at low frequencies), the detector reaches an ROC-AUC
  of about **0.93** — good, and deliberately *not* a suspiciously perfect score.
- On **real recorded audio**, a small neural network reaches **~0.99 AUC** and
  detects **~94%** of drones at a strict **1%** false-alarm rate — but only after
  we removed a subtle data-handling mistake that had inflated the numbers (see
  the [leakage chapter](06_evaluation_rigor_and_leakage.md)).
- For **direction-finding**, the array locates the bearing to within a fraction
  of a degree at healthy signal levels, degrading gracefully as noise rises.
- The **blade-passing frequency** is recovered almost exactly (e.g. 110 Hz true
  → 109 Hz estimated).

## Honest limitations

EchoHawk is a demonstrator. The real-audio test uses "easy" negatives (everyday
sounds, not hard confusers like lawnmowers); the direction-finding results are
mostly on simulated arrays (the real-array script is ready but the DREGON data
is pending); and it is single-source, far-field. These are spelled out in the
[evaluation chapter](06_evaluation_rigor_and_leakage.md) and the project README.
The point of the project is **correct method and honest measurement**, not a
headline number.
