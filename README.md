# Multi-View EEGNet with Domain-Adversarial and Contrastive Learning

A cross-subject motor imagery EEG classification pipeline for the BCI Competition IV 2a and PhysioNet EEG Motor Movement/Imagery datasets. The project implements a lightweight Multi-View EEGNet architecture with time-domain, frequency-domain, and segmented band-power views, combined with temporal cross-attention, domain-adversarial training, supervised contrastive learning, and robust inference-time calibration.

## Overview

This repository contains two related EEG motor imagery classification pipelines:

- `bci/`: BCI Competition IV Dataset 2a, 4-class motor imagery classification
- `CONVOPHYSIO/`: PhysioNet EEG Motor Movement/Imagery, binary left-vs-right hand motor imagery classification

The main goal is cross-subject generalization, where the model must classify motor imagery trials from unseen subjects.

## Key Features

- Multi-view EEG representation:
  - Raw time-domain EEG
  - Log-power frequency-domain features
  - Temporally segmented mu/beta band-power features

- Modified EEGNet backbone:
  - Factorized temporal and spatial convolutions
  - Depthwise and separable convolution blocks
  - Lightweight parameter count

- Temporal cross-attention:
  - Aligns segmented band-power features with time/frequency features
  - Uses a learned query blending parameter

- Branch fusion:
  - Learns soft-attention weights across time, frequency, and band-power branches

- Cross-subject regularization:
  - Domain-Adversarial Neural Network head
  - Supervised Contrastive Learning head
  - R-Drop consistency regularization
  - Label smoothing
  - Strong dropout and weight decay

- Robust inference:
  - Exponential Moving Average evaluation
  - Adaptive Batch Normalization
  - Test-Time Augmentation
  - MC-Dropout
  - Temperature scaling

## Repository Structure

```text
.
├── bci/
│   ├── eeg_preprocessing_core.py
│   ├── multiview_features.py
│   ├── phase2.py
│   ├── evaluation.py
│   ├── train.py
│   ├── bcirun.txt
│   └── bciresults/
│       ├── v16bci_preparedf/
│       │   └── prepared_meta.json
│       └── v24bci_outputf/
│           ├── aggregate_summary.csv
│           ├── aggregate_summary.json
│           ├── fold_metrics.csv
│           └── fold_*/
│               ├── confusion_matrix.png
│               └── fold_metrics.json
│
├── CONVOPHYSIO/
│   ├── PHYSIOeeg_preprocessing_core.py
│   ├── PHYSIOmultiview_features.py
│   ├── PHYSIOphase2.py
│   ├── PHYSIOevaluation.py
│   ├── PHYSIOtrain.py
│   ├── RESULTSLOG.TXT
│   └── RESULTSPHYSIO/
│       ├── 122pipelineOP/
│       │   ├── aggregate_summary.csv
│       │   ├── aggregate_summary.json
│       │   ├── fold_metrics.csv
│       │   └── fold_*/
│       │       ├── confusion_matrix.png
│       │       └── fold_metrics.json
│       └── 122preprocessed/
│           └── prepared_meta.json
│
├── .gitignore
└── .gitattributes
```

## Datasets

### BCI Competition IV Dataset 2a

| Property | Value |
|---|---|
| Subjects | 9 |
| Classes | 4 |
| Classes Used | Left hand, right hand, feet, tongue |
| EEG Channels | 22 |
| Sampling Rate | 250 Hz |
| Sessions | 2 per subject |
| Trials per Session | 288 |
| Trials Used | 2,592 training-session trials |
| Protocol | Leave-Two-Subjects-Out cross-validation |

The evaluation sessions are not used because labels are unavailable in the local pipeline setup.

### PhysioNet EEG Motor Movement/Imagery

| Property | Value |
|---|---|
| Subjects | 109 total, 107 usable |
| Classes | 2 |
| Classes Used | Left hand vs right hand motor imagery |
| EEG Channels | 64 |
| Sampling Rate | 160 Hz |
| Runs Used | R03, R07, R11 |
| Trials per Subject | Approximately 90 |
| Trials Used | Approximately 9,630 |
| Protocol | Subject-level 5-fold cross-validation |

Subjects S088 and S092 are skipped because of short or incompatible trials.

## Preprocessing

### BCI-IV 2a

- EOG channels are discarded.
- EEG is bandpass filtered from 0.5 Hz to 30 Hz.
- A 2-second trial window is extracted.
- The window starts 0.5 seconds after cue onset to reduce cue-preparation effects.
- Euclidean Alignment is applied per subject.
- Channel normalization is performed inside the model.

### PhysioNet

- EEG is bandpass filtered from 4 Hz to 38 Hz.
- A 2-second trial window is extracted.
- Subject-wise Euclidean Alignment is applied.
- Channel normalization is performed inside the model.

## Multi-View Feature Extraction

Each trial is converted into three complementary views.

### Time View

Raw filtered EEG.

```text
BCI:       N x 22 x 500
PhysioNet: N x 64 x 320
```

This view preserves temporal morphology and event-related desynchronization/synchronization patterns.

### Frequency View

Log-power spectral density computed from the real FFT.

```text
power = abs(rfft(x))^2
freq = 10 * log10(power + 1e-12)
```

This view captures mu and beta spectral distributions.

### Segmented Band-Power View

Each trial is divided into 8 temporal segments. For each segment, log-bandpower is computed for:

- Mu band: 8-12 Hz
- Beta band: 13-30 Hz

This produces 16 features per channel.

The segmented band-power view captures when spectral changes occur inside the trial, instead of only measuring static trial-level bandpower.

## Model Architecture

The model is a Multi-View EEGNet with three input branches.

```text
Input: time, frequency, segmented band-power
  |
ChannelNorm
  |
EEGNet Stem for time view
EEGNet Stem for frequency view
Band Stem for segmented band-power view
  |
Multi-Scale Branches
  |
Temporal Cross-Attention
  |
Learned Branch Fusion
  |
Squeeze-and-Excitation Block
  |
Temporal Convolutional Network Stack
  |
Temporal Attention
  |
Adaptive Average Pooling
  |
Pre-classifier
  |
Classifier
```

Auxiliary heads are used during training:

```text
Pre-classifier features
  |
  |-- Domain-Adversarial subject classifier
  |
  |-- Supervised Contrastive projection head
```

## Main Architectural Components

### EEGNet Stem

The time and frequency views use modified EEGNet-style factorized convolution blocks:

- Temporal convolution
- Depthwise spatial convolution
- Separable temporal convolution
- Batch normalization
- ELU activation
- Average pooling
- Dropout

### Multi-Scale Branch

Each branch uses parallel temporal convolutions with different kernel sizes:

```text
Conv1d kernel 3
Conv1d kernel 7
Conv1d kernel 15
```

The outputs are concatenated and projected into a shared branch representation.

### Temporal Cross-Attention

The segmented band-power branch is aligned with the time/frequency branches using cross-attention.

A learned query blending parameter controls how much the attention query depends on the time branch versus the frequency branch.

### Branch Fusion

The model learns soft weights for each branch:

```text
fused = w_time * time + w_freq * freq + w_band * band
```

The weights are produced by a small branch-scoring network and normalized with softmax.

### Domain-Adversarial Training

A Gradient Reversal Layer is used to train the feature extractor against a subject classifier. This encourages subject-invariant representations.

### Supervised Contrastive Learning

A projection head maps features to a normalized embedding space. Samples from the same class are pulled together, while samples from different classes are pushed apart.

## Training Objective

The total training loss is:

```text
L_total = L_CE + lambda_DANN * L_DANN + w_SupCon * L_SupCon + alpha_RDrop * L_RDrop
```

Where:

- `L_CE`: Cross-entropy or focal cross-entropy
- `L_DANN`: Subject classification loss through gradient reversal
- `L_SupCon`: Supervised contrastive loss
- `L_RDrop`: KL consistency loss between two stochastic forward passes

## Training Configuration

| Setting | BCI-IV 2a | PhysioNet |
|---|---:|---:|
| Optimizer | AdamW | AdamW |
| Backbone LR | 3e-4 | 3e-4 |
| Head LR | 1.5e-4 | 1.5e-4 |
| Weight Decay | 0.08 | 0.02 |
| Scheduler | Cosine annealing | Cosine annealing |
| Batch Size | 32 | 32 |
| Mixed Precision | Yes | Yes |
| Main Dropout | 0.50-0.60 | 0.50-0.60 |
| DANN Weight | 0.03-0.05 | 0.03-0.05 |
| SupCon Weight | 0.05 | 0.05 |

## Data Augmentation

Online augmentations include:

- Gaussian noise
- Time masking
- Channel dropout
- Temporal shift
- Amplitude scaling

Augmentation strength is ramped up during early training to avoid instability.

## Results

### BCI Competition IV Dataset 2a

| Metric | Value |
|---|---:|
| Test Accuracy | 55.71 percent ± 9.07 percent |
| Test Macro AUC | 0.7945 ± 0.0670 |
| Test Macro F1 | 0.5548 ± 0.0899 |
| Expected Calibration Error | 0.0992 ± 0.0379 |

The BCI pipeline uses a strict Leave-Two-Subjects-Out protocol: one subject is held out for validation and another for testing.

### PhysioNet EEG Motor Movement/Imagery

| Metric | Value |
|---|---:|
| Test Accuracy | 76.10 percent ± 0.97 percent |
| Test Macro AUC | 0.9070 ± 0.0055 |
| Test Macro F1 | 0.7369 ± 0.0087 |
| Expected Calibration Error | 0.0329 |

The PhysioNet pipeline uses subject-level 5-fold cross-validation.

## How to Run

### 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas scikit-learn matplotlib seaborn mne tqdm
```

Depending on your machine and CUDA version, install PyTorch from the official PyTorch instructions.

### 3. Run BCI Pipeline

```bash
cd bci
python train.py
```

The BCI pipeline expects the BCI-IV 2a dataset to be available in the paths configured inside the preprocessing and training scripts.

### 4. Run PhysioNet Pipeline

```bash
cd CONVOPHYSIO
python train.py
```

The PhysioNet pipeline expects the PhysioNet EEG Motor Movement/Imagery dataset to be available in the configured data directory.

## Output Files

Typical output files include:

```text
aggregate_summary.csv
aggregate_summary.json
fold_metrics.csv
fold_*/fold_metrics.json
fold_*/confusion_matrix.png
```

Large generated artifacts such as preprocessed arrays, model checkpoints, and compressed result archives may be regenerated locally from the scripts.

## Notes on Large Files

Large binary files such as `.npy`, `.pt`, and `.zip` artifacts are not required to inspect the codebase and may exceed normal GitHub repository limits. These files should be regenerated locally or stored separately using Git LFS, cloud storage, or a dataset release system.

## Novelty

This project contributes a compact cross-subject EEG motor imagery framework with the following components:

1. A multi-view EEG representation combining raw time-domain, frequency-domain, and segmented band-power features.

2. A segmented band-power view that captures time-varying mu and beta energy instead of static trial-level bandpower.

3. A temporal cross-attention mechanism that dynamically aligns band-power features with discriminative temporal positions.

4. A learned branch-fusion module that adaptively weights time, frequency, and band-power branches.

5. Joint domain-adversarial and supervised contrastive training for subject-invariant but class-discriminative feature learning.

6. A robust inference pipeline using calibration, test-time augmentation, MC-Dropout, and adaptive batch normalization.

## Key References

- Brunner, C., et al. BCI Competition 2008 - Graz data set A.
- Schalk, G., et al. BCI2000: A General-Purpose Brain-Computer Interface System.
- Lawhern, V. J., et al. EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
- Ganin, Y., et al. Domain-Adversarial Training of Neural Networks.
- Khosla, P., et al. Supervised Contrastive Learning.
- He, H., and Wu, D. Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach.
- Liang, X., et al. R-Drop: Regularized Dropout for Neural Networks.
- Hu, J., et al. Squeeze-and-Excitation Networks.
- Bai, S., et al. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.
- Guo, C., et al. On Calibration of Modern Neural Networks.


## License

Copyright (c) 2026 ASHWINDER PAL SINGH. All Rights Reserved. This code may not be copied, reproduced, or distributed without explicit permission.
