# 2. Concepts primer — the ideas, explained simply

This chapter explains every idea EchoHawk uses, assuming no background. Each
section builds on the last. Terms in **bold** are in the [glossary](09_glossary.md).

## 2.1 Why a drone is "loud" in a very particular way

A multirotor drone has propellers spinning thousands of times per minute. Each
time a blade slices the air, it makes a little pressure pulse. Because the blades
pass at a steady rate, those pulses repeat at a fixed frequency — the
**blade-passing frequency (BPF)**.

A repeating sound is not just one tone; it comes with **harmonics** — extra tones
at exact multiples of the fundamental (2×, 3×, 4× …). Together they form a
**harmonic stack**: a neat ladder of evenly spaced tones. A guitar string,
a human voice, and a drone all produce harmonic stacks; the *spacing and shape*
of the ladder is what makes each sound distinctive.

> **Key idea:** A drone's acoustic "fingerprint" is a harmonic stack starting at
> its blade-passing frequency (often a few hundred hertz), sitting on top of a
> broadband "whoosh" of rotor turbulence.

The hard part: **ground vehicles and generators also hum with low-frequency
harmonics.** Telling a drone from a truck is the real challenge — which is why
EchoHawk's synthetic test deliberately includes vehicle-like "confusers".

## 2.2 Turning sound into a picture: the spectrogram

Sound is a wiggle of air pressure over time. That raw wiggle is hard to read. So
we chop the sound into short slices and, for each slice, ask "how much energy is
at each frequency?" using a mathematical tool called the **Fourier transform**.
Stacking those slices side by side gives a **spectrogram**: a picture with time
across the bottom, frequency up the side, and brightness showing loudness.

On a spectrogram, a drone's harmonic stack shows up as a set of **bright,
evenly-spaced horizontal lines**. This visual regularity is exactly what a
classifier learns to recognise.

A **log-mel spectrogram** is the same picture with the frequency axis warped to
match human hearing (the "mel" scale) and brightness shown on a log scale —
a representation that works very well as input to neural networks. **MFCCs**
(Mel-Frequency Cepstral Coefficients) are a compact numerical summary of that
picture, handy for simpler classifiers.

## 2.3 Detecting the fingerprint: the blade-passing frequency

To estimate the BPF, EchoHawk uses the **Harmonic Product Spectrum (HPS)**: it
squashes the spectrum by factors of 2, 3, 4 … and multiplies the copies
together. Wherever a true fundamental and its harmonics line up, the product
spikes — pinpointing the BPF even in noise. (In the demo, 110 Hz true → 109 Hz
estimated.)

## 2.4 Deciding "drone or not": classification

A **classifier** is a function that takes features (numbers describing a clip)
and outputs a label. EchoHawk uses two:

- A **Random Forest** — a committee of decision trees voting — on hand-crafted
  features (MFCC summaries, harmonic cues). Simple, fast, a good *baseline*.
- A small **Convolutional Neural Network (CNN)** that looks at the whole
  spectrogram picture and learns the pattern itself. More powerful.

How good is a detector? Two numbers matter:

- **Detection rate (Pd)** — of the real drones, what fraction did we catch?
- **False-alarm rate (Pfa)** — of the non-drones, what fraction did we wrongly
  flag? In security, false alarms are expensive, so we care about **Pd at a low
  Pfa** (e.g. "what % of drones do we catch while crying wolf only 1% of the
  time?"). The **ROC curve** plots Pd against Pfa across all thresholds; its area
  (**AUC**) is a single summary, where 1.0 is perfect and 0.5 is a coin flip.

## 2.5 Finding the direction: microphone arrays

One microphone tells you *that* a drone is there, not *where*. Two ears let you
localise sound because the wave reaches the nearer ear a hair earlier. Use
several microphones in a known geometry — an **array** — and those tiny
**time differences** reveal the **direction of arrival (DOA)**, the bearing to
the source.

A few physical facts drive the design:

- Sound travels at about **343 m/s**. A 500 Hz tone has a **wavelength** of about
  0.69 m (wavelength = speed ÷ frequency).
- Microphones must sit **no more than half a wavelength apart**, or directions
  become ambiguous (like a clock with no hour hand).
- A **bigger array** gives sharper direction resolution.

## 2.6 Listening in a direction: beamforming

**Beamforming** combines the microphone signals so the array "listens" most
sensitively in one chosen direction. Sweep the chosen direction across all angles
and the angle with the most energy is your estimate. EchoHawk implements three,
from simple to sharp:

- **Delay-and-Sum (Bartlett):** line the signals up and add them. Robust, but a
  wide, blurry beam.
- **MVDR / Capon:** actively *suppress* energy from other directions while
  keeping the chosen one — a sharper beam.
- **MUSIC:** a clever method using the mathematical structure of the signals to
  produce very sharp, high-resolution peaks.

A complementary idea is **GCC-PHAT**, which measures the precise time delay
between a pair of microphones, and **SRP-PHAT**, which combines all pairs into a
robust direction map. Several spaced microphone *nodes* let you **triangulate**
the actual position, not just the bearing — the principle behind real systems
that use a handful of networked sensors.

## 2.7 Following it over time: tracking

A single estimate is noisy. Because a real drone moves smoothly, we can **track**
it: a **tracker** (here a simple Kalman-style smoother) blends each new noisy
bearing with the predicted motion, producing a clean trajectory. In the demo this
cuts the bearing error from ~5.7° (raw) to ~1.9° (tracked).

## 2.8 Signal-to-noise ratio (SNR)

Throughout, performance depends on **SNR** — how strong the drone signal is
compared with background noise, in decibels (dB). High SNR (loud, close drone) is
easy; low or negative SNR (faint, distant, windy) is hard. EchoHawk reports
results *as a function of SNR* so you can see exactly how gracefully each method
degrades — that honesty is more useful than a single best-case figure.

➡ Next: [System architecture](03_architecture.md) — how these ideas are wired
together into a pipeline.
