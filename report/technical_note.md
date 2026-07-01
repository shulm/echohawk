# EchoHawk: A Passive Acoustic System for Drone Detection, Classification, and Direction-Finding

**David Shulman, Ph.D.**

## Abstract
The rapid proliferation of Unmanned Aerial Vehicles (UAVs) presents significant security challenges, necessitating robust, passive counter-UAS (C-UAS) systems. This technical note details EchoHawk, an acoustic-based pipeline for detecting, classifying, and tracking drones. The system exploits the distinct harmonic signature of UAV rotors (blade-passing frequency) to differentiate drones from hard confusers such as ground vehicles. We implement and evaluate traditional machine learning models (Random Forests over MFCCs) alongside Convolutional Neural Networks (CNNs) operating on log-mel spectrograms. Crucially, we demonstrate the severe impact of data leakage when evaluating on public datasets (such as the DroneAudioDataset) and present corrected metrics using rigorous session-level grouping. Finally, we incorporate wideband array signal processing techniques (MUSIC, MVDR, SRP-PHAT) coupled with a Kalman tracker for continuous Direction of Arrival (DOA) estimation.

## 1. Introduction
Passive acoustic monitoring is a critical modality for drone detection, offering advantages over radar (which struggles with low cross-sections) and optical systems (which require line-of-sight and good visibility) [1]. Drones generate a highly structured acoustic signature composed of a fundamental blade-passing frequency (BPF) and its integer harmonics. 

However, detecting this signature is complicated by environmental noise and "hard confusers"—such as ground vehicles and generators—that produce similar low-frequency harmonic noise. Furthermore, localising the source in real-time requires robust microphone array processing. The EchoHawk system integrates detection, feature extraction (BPF), and localisation into a unified, reproducible framework.

## 2. Methods

### 2.1 Detection and Classification
The detection pipeline frames the task as binary classification: drone vs. non-drone. Two primary architectures were evaluated:
1. **RF Baseline**: A Random Forest classifier (300 trees) operating on hand-crafted features, primarily Mel-Frequency Cepstral Coefficients (MFCCs) and harmonic markers like Spectral Flatness and Harmonic Product Spectrum (HPS).
2. **DroneCNN**: A lightweight PyTorch Convolutional Neural Network operating directly on 1-second log-mel spectrograms. The CNN leverages 2D convolutions to learn the spatial-temporal structure of the harmonic stack, including the subtle RPM wobble characteristic of multi-rotor UAVs.

### 2.2 Direction of Arrival (DOA)
For spatial tracking, EchoHawk employs far-field wideband beamforming over a planar microphone array. The system processes multi-channel audio via Short-Time Fourier Transform (STFT) and aggregates narrowband spatial covariance matrices across the dominant harmonic band (e.g., 80–4000 Hz).
Three primary azimuth estimators are implemented:
- **Bartlett (Delay-and-Sum)**: Maximises projected power.
- **MVDR (Capon)**: Minimises total variance while maintaining unity gain in the look direction.
- **MUSIC**: A subspace method that projects steering vectors onto the noise subspace for high-resolution localisation.
Additionally, we implement SRP-PHAT, accumulating GCC-PHAT pairwise time delays into a robust azimuth power map. Tracking is stabilized using a 1-D $\alpha-\beta$ tracker to smooth noisy sequential measurements.

## 3. Datasets

### 3.1 Synthetic Benchmark
To rigorously evaluate the algorithms under controlled conditions, we constructed a physically transparent simulator. The drone is modelled as a harmonic stack embedded in noise, while negatives include exact confusers (ground vehicles with different harmonic decay and FM modulation) and background noise (pink/white). The simulator also generates multi-channel array captures using a plane-wave propagation model.

### 3.2 DroneAudioDataset (Real Data)
We evaluated the models on the public DroneAudioDataset [3], containing drone recordings and general environmental/speech sounds. A critical finding during our evaluation was the presence of severe session-level data leakage. The dataset's 1,332 drone audio files originate from only 257 continuous recording sessions. Because the files were pre-segmented into 1-second clips, standard file-level randomized splits inadvertently placed adjacent clips from the same session into both the training and testing sets, allowing the model to learn background acoustic characteristics rather than the drone signature. We corrected this by enforcing a strict session-level `GroupShuffleSplit`.

## 4. Results

### 4.1 Synthetic Benchmark
On the synthetic detection task (5-fold CV, balanced classes, SNR spanning -15 to +3 dB), the baseline achieved an ROC-AUC of 0.93 and an accuracy of 0.86. The DOA estimators were evaluated on a simulated 8-microphone uniform circular array (radius = 10 cm). MUSIC and MVDR exhibited sub-degree Root Mean Square Error (RMSE) at 0 dB SNR, significantly outperforming the Bartlett beamformer.

| Array SNR | Bartlett | MVDR | MUSIC |
|---|---|---|---|
| −10 dB | ~4° | ~1.6° | ~2.3° |
| 0 dB | ~0.6° | ~0.8° | ~0.9° |
| +10 dB | ~0.3° | ~0.3° | ~0.3° |

### 4.2 Real Data (DroneAudioDataset)
Table 2 contrasts the inflated file-grouped metrics with the honest, session-grouped metrics. Under strict session-level cross-validation, the RF baseline suffered a 5.1% drop in Probability of Detection (Pd) at a 1% False Alarm (FA) rate. The CNN retained strong performance, indicating that its deep temporal receptive field generalizes better to unseen environments.

*Table 2: Real-data Detection Performance (File-level vs. Session-level Grouping)*

| Metric | RF (File-grouped) | RF (Session-grouped) | CNN (File-grouped) | CNN (Session-grouped) |
|--------|-------------------|----------------------|--------------------|-----------------------|
| **ROC-AUC** | 0.9872 | **0.9815** | | 0.9968 | **0.9941** |
| **Accuracy**| 0.9631 | **0.9638** | | 0.9868 | **0.9840** |
| **Pd@1%FA** | 0.7964 | **0.7453** | | 0.9519 | **0.9384** |
| **Pd@5%FA** | 0.9446 | **0.9185** | | 0.9650 | **0.9784** |
| **Pd@10%FA**| 0.9745 | **0.9499** | | 0.9971 | **0.9834** |

## 5. Limitations
The real-data evaluation on the DroneAudioDataset is inherently an "easy-negative" benchmark, as the negative class comprises ESC-50 sounds rather than operational confusers like lawnmowers. These hard confusers are currently only evaluated in the synthetic benchmark. Additionally, the real-world DOA pipeline expects data from the DREGON dataset [2], but physical array recordings remain pending local network access limits.

## 6. Future Work
Future extensions include training the DroneCNN on larger aggregations of datasets such as DADS [4] and validating the end-to-end DOA tracking pipeline on physical drone flight recordings (e.g., DREGON) to characterize multi-path fading and wind noise effects on the GCC-PHAT estimator.

## 7. References
[1] Busset, J., et al. "Detection and tracking of drones using advanced acoustic cameras." *Proc. SPIE*, 2015.  
[2] Strauß, M., et al. "DREGON: Dataset and Methods for UAV-Embedded Sound Source Localization." *IROS*, 2018.  
[3] Al-Emadi, S., et al. "Audio Based Drone Detection and Identification using Deep Learning Techniques." *IEEE*, 2019.  
[4] "Drone Acoustic Detection System (DADS) Dataset." *Hugging Face*, 2023.
