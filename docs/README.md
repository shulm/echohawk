# EchoHawk Documentation

Welcome. EchoHawk is a small, open, **acoustic drone-detection** system: it
listens for the sound a drone makes, decides whether a drone is present,
estimates the direction the sound is coming from, and tracks it over time.

This documentation is written so that **someone with no signal-processing or
machine-learning background can follow the whole project** — while still being
precise enough for an engineer or reviewer.

## How to read these docs

**If you are new to the topic**, read in order:

1. [Overview](01_overview.md) — what problem this solves and what EchoHawk does, in plain language.
2. [Concepts primer](02_concepts_primer.md) — the ideas behind it (drone sound, spectrograms, microphone arrays, direction-finding) explained simply, with analogies.
3. [System architecture](03_architecture.md) — how the pieces fit together.

**If you want the technical substance:**

4. [Detection & classification](04_detection_and_classification.md) — how EchoHawk decides "drone or not", and the results.
5. [Direction-finding & tracking](05_direction_finding_and_tracking.md) — how it finds and follows the bearing.
6. [Evaluation, rigor & the leakage story](06_evaluation_rigor_and_leakage.md) — how we made sure the numbers are honest (the most important chapter).

**If you want to run or extend it:**

7. [Getting started](07_getting_started.md) — install and run every demo.
8. [API reference](08_api_reference.md) — every public function.
9. [Glossary](09_glossary.md) — plain-language definitions of all the jargon.

## One-paragraph summary

A drone's spinning propellers create a very regular, layered hum (a *harmonic
stack*). EchoHawk turns recorded sound into a picture (a *spectrogram*),
recognises that drone-like pattern with a classifier (a Random Forest baseline
and a small neural network), and — using several microphones — works out the
**direction** the sound came from using classic array-processing methods
(delay-and-sum, MVDR, MUSIC) and time-delay triangulation (GCC-PHAT / SRP-PHAT),
finally smoothing the bearing over time with a tracker. Everything runs on
built-in **synthetic data**, so you can try it without downloading anything, and
it has been checked against **real recordings** with careful, leakage-free
evaluation.

See the [main project README](../README.md) for headline results and the
[technical note](../report/technical_note.md) for the paper-style write-up.
