# India Air Quality Data — GAN-Based Density Estimation

## Overview

This notebook trains a **Generative Adversarial Network (GAN)** to learn and replicate the distribution of a transformed NO₂ (Nitrogen Dioxide) signal derived from Indian air quality data. After training, it uses **Kernel Density Estimation (KDE)** to visualize the probability density of the GAN-generated samples.

---

## Dataset

- **Source:** [India Air Quality Data](https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data) (Kaggle)
- **File:** `data.csv`
- **Feature used:** `no2` — Nitrogen Dioxide concentration readings (µg/m³), cleaned of NaN values and converted to a PyTorch tensor.

---

## Data Transformation

Raw NO₂ values (`x`) are transformed into a new signal `z_real` using a **sinusoidal function** based on a student registration number `r`:

```
r     = 102353012
a_r   = 0.5 × (r mod 7)   →  0.5 × 5 = 2.5
b_r   = 0.3 × (r mod 5 + 1) →  0.3 × 3  = 0.9

z_real = a_r × sin(b_r × x)
       = 2.5 × sin(0.9 × x)
```

> **Note:** Unlike Assignment 3 where `z = x + perturbation`, here `z_real` is the sinusoidal component alone — a non-linear compression of the NO₂ values into the range **[-2.5, 2.5]** (since sine is bounded by ±1, scaled by `a_r = 2.5`).

---

## Model Architecture

### Generator (G)
Takes random Gaussian noise as input and learns to produce samples that mimic the distribution of `z_real`.

```
Input (1) → Linear(1→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1) → Output (1)
```

### Discriminator (D)
Takes a sample (real or fake) and outputs a probability of it being real.

```
Input (1) → Linear(1→32) → ReLU → Linear(32→1) → Sigmoid → Output (probability)
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| **Epochs** | 2000 |
| **Batch size** | 64 |
| **Loss function** | Binary Cross-Entropy (`BCELoss`) |
| **Optimizer (G & D)** | Adam, learning rate = 0.001 |

### Training Loop (per epoch):
1. Sample a random batch of real `z_real` values.
2. Generate fake samples from Gaussian noise via Generator `G`.
3. **Train Discriminator:** Minimize loss on real samples (label = 1) + fake samples (label = 0).
4. **Train Generator:** Minimize loss trying to fool the Discriminator (label fake as 1).

---

## Post-Training: KDE Estimation

After training, **2000 samples** are generated from the trained Generator using random noise. A **Gaussian Kernel Density Estimator (KDE)** with bandwidth `0.3` is fitted on these generated samples to produce a smooth PDF curve.

```
kde = KernelDensity(kernel="gaussian", bandwidth=0.3)
kde.fit(z_fake)
```

---

## Graph: GAN-Generated Histogram + KDE PDF

The output plot (`gan_pdf_plot.png`) shows:

- **X-axis:** Values of the GAN-generated transformed signal `z`
- **Y-axis:** Density
- **Blue bars (histogram):** Distribution of 2000 samples produced by the trained Generator (40 bins, normalized to density)
- **Orange curve (KDE PDF):** Smooth probability density estimated over the generated samples

### What the graph shows:
The GAN has learned to generate samples resembling the sinusoidally transformed NO₂ distribution. Since `z_real = 2.5 × sin(0.9 × x)` maps all values into **[-2.5, 2.5]**, the generated distribution should cluster within this bounded range. The KDE curve smooths out the histogram to reveal the underlying density shape the GAN has learned.

---

## Key Difference from Assignment 3

| Aspect | Assignment 3 | Assignment 4 |
|---|---|---|
| **Method** | Analytical Gaussian fit | GAN + KDE |
| **Transform** | `z = x + 0.25×sin(0.9x)` | `z = 2.5×sin(0.9x)` |
| **Output range** | Similar to NO₂ values (~0–200) | Bounded to [-2.5, 2.5] |
| **PDF estimation** | Closed-form Normal distribution | Data-driven (learned by GAN, smoothed by KDE) |

---

## Dependencies

```
numpy
pandas
torch
torch.nn
torch.optim
matplotlib
scikit-learn (KernelDensity)
kagglehub
```

---

## How to Run

1. Ensure you have a Kaggle API key configured for `kagglehub`.
2. Install dependencies:
   ```
   pip install numpy pandas torch matplotlib scikit-learn kagglehub
   ```
3. Run all cells sequentially in the notebook.
4. The output plot is saved as `gan_pdf_plot.png`.
