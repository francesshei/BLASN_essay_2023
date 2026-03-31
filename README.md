# Multiscale Sample Entropy Analysis of EEG Data in Alzheimer's Disease and Frontotemporal Dementia
A signal complexity analysis of resting-state EEG recordings from  healthy controls, Alzheimer's Disease (AD), and Frontotemporal Dementia (FTD) patients, using Multiscale Sample Entropy (MSE) as  a non-linear complexity measure.

Written as a final essay for the *Basic Linear Algebra and Statistics for Neuroscience* course (National PhD in AI, XVII cycle). 
Full write-up included alongside the code.

---

## What this project does

EEG complexity (how irregular and information-rich the signal is) is known to differ between healthy and pathological brains. This project asks: **where** in the scalp and **at what temporal scales** do these differences appear for AD and FTD?

Starting from raw EEG recordings (19 electrodes, 500 Hz, 88 subjects), the pipeline:

1. Preprocesses raw signals using MNE-Python (bandpass filter  0.5–45 Hz, epoch segmentation, artefact rejection)
2. Computes MSE for each subject at each electrode site across 20 temporal scale factors
3. Tests for statistical differences across groups (AD / FTD / healthy control) using Kruskal-Wallis tests at each electrode-scale combination (380 tests total)
4. Visualises results spatially against the 10-20 EEG topographic map

**Key finding:** statistically significant, spatially-localised complexity reductions in posterior-occipital electrodes (O1, O2, T5, P3) in both pathological groups vs. healthy controls. This is consistent with Default Mode Network disruption literature in AD.

---

## Dataset

Public dataset from OpenNeuro:  [ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.2)  
88 subjects: 36 AD, 23 FTD, 29 healthy controls; 19-channel scalp EEG, 10-20 system, 500 Hz sampling rate

---

## Stack

- **Python** — MNE-Python (preprocessing), NumPy, SciPy, Matplotlib
- **R / RStudio** — statistical analysis (Bartlett's test, Kruskal-Wallis), visualisation
- **MSE implementation** — custom Python, validated against Costa et al. (2005) on white and pink noise benchmarks

---

## Essay

More information, figures and references can be found in `essay.pdf`
