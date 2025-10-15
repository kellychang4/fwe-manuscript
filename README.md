
# FWE Manuscript

This repository contains the scripts used to preprocess, analyze, and visualize the data found in: 

Chang, K., Burke, L., LaPiana, N., Howlett, B., Hunt, D., Dezelar, M., Andre, J. B., Curl, P., Ralston, J. D., Rokem, A., & Mac Donald, C. L. (2024). Free water elimination tractometry for aging brains. bioRxiv.
[https://doi.org/10.1101/2024.11.10.622861](https://doi.org/10.1101/2024.11.10.622861)

[!NOTE] All scripts have been edited to remove participant identifiers and local/server path information.

## Repository Organziation

The repository is organized into the following subdirectories, ordered generally by the oprder in which the processing pipeline occured.

- [`1_preprocess/`](#1_preprocess)

  This directory contains scripts used for preprocessing. Scripts were considered as preprocessing if they primarily interacted with the raw datasets.

- [`2_analysis/`](#2_analysis)

  This directory contains scripts used for analysis. Scripts were considered as analysis if they primarily interacted with the derivative outputs generated during the preprocessing step.

- [`3_figures/`](#3_figures)

  This directory contains scripts used for manuscript figure creation.

- [`4_helper/`](#4_helper)

  This directory contains helper scripts.

---

### `1_preprocess/`

Preprocessing was performed on a high-performance computing (HPC) server managed by a [Slurm](https://slurm.schedmd.com/overview.html) workload scheduler. Each preprocessing step consisted of a paired Python script (`.py`) and job script (`.job`).

The typical workflow involved submitting the job through the job script. This script launched an [Apptainer](https://apptainer.org/) container that contained a Python environment and executed the corresponding Python script for that preprocessing step.

#### Preprocessing

1. `run_qsiprep.[py|job]`: Runs [QSIprep](https://qsiprep.readthedocs.io/en/latest/index.html) preprocessing diffusion MRI data. 
2. `run_hypermapper.[py|job]`: Run [HyperMapp3r](https://hypermapp3r.readthedocs.io/en/latest/) preprocessing for segmenting white matter hyperintensities from FLAIR images.

#### Diffusion Modeling

1. `fit_dki.[py|job]`: Fit the [Diffusion Kurtosis Imaging (DKI) model with the White Matter Tract Integrity (WMTI) technique](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki_micro.html#reconstruction-of-the-diffusion-signal-with-the-wmti-model-dki-micro) to the preprocessed diffusion data.
2. `fit_fwdti.[py|job]`: Fit the [Free Water Diffusion Tensor Imaging (FWDTI) model](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_fwdti.html#using-the-free-water-elimination-model-to-remove-dti-free-water-contamination) to the preprocessed diffusion data.
3. `fit_msmt.[py|job]`: Fit the [Multi-Shell Multi-Tissue (MSMT) Constrained Spherical Deconvoukltion (CSD)](https://docs.dipy.org/stable/examples_built/reconstruction/reconst_mcsd.html#reconstruction-with-multi-shell-multi-tissue-csd) model to the preprocessed diffusion data. 

#### Tractometry

1. `prepare_afq.[py|job]`: Prepares preprocessed dataset for pyAFQ by creating white matter masks and performing free water elimination to the diffusion data. 
2. `run_afq_original.[py|job]`: Run [pyAFQ](https://tractometry.org/pyAFQ/) on the unmodified diffusion dataset (condition name: original). 
3. `run_afq_fwe.[py|job]`: Run [pyAFQ](https://tractometry.org/pyAFQ/) on the free water eliminated diffusion dataset (condition name: fwe). 
4. `run_afq_msmt.[py|job]`: Run [pyAFQ](https://tractometry.org/pyAFQ/) on the MSMT estimated white matter fiber orientation distribution functions (condition name: msmt). 
5. `run_tract_profiles.[py.job]`: Calculates the tract profiles for each tract, diffusion metric, and method.

#### Within-Participant Split-Half Reliability

1. `run_reliability.[py|job]`: Runs the within-participant split-half reliability pipeline.

  This pipeline first divides the raw multi-shell diffusion dataset into two subsets ("splits") by randomly and evenly distributing the b-values across splits. Each split is then processed independently through the same steps described in the [Diffusion Modeling](#diffusion-modeling) and [Tractometry](#tractometry-with-pyafq) sections.

---

### `2_analysis/`

#### Quality Control

- `create_quality_control.py`: Creates a list of participants that could be retained due to their quality control metrics.
  
#### Compiling Scripts

Compiling scripts were used to collate derived outputs across participants. These scripts write a CSV files containing the requested information.

- `create_demographics.py`: Create a demographics CSV file.
- `create_fazekas_score.py`: Create a Fazekas score CSV file.
- `create_tract_stats.py`: Create a tract statistics (streamline count, median tract length) CSV file. 
- `create_profiles.py`: Create a tract profile CSV file. 
- `create_wmh_profiles.py`: Create a tract profile CSV for the interaction of tract with white matter hyperintensities (WMH). 

#### Computation Scripts

Computation scripts calculated reliability metrics and summary statistics from the split-half dataset.

- `calc_fodf_corr2.py`: Calculates the squared correlation ($r^{2}$) between fiber orientation distribution functions (fODFs) across split-halves for each method.
- `calc_tract_dice.py`: Calculates the Dice coefficient of tract segmentations across split-halves for each tract and method.
- `calc_profile_icc.py`: Calculates intraclass correlation coefficients (ICC) of tract profiles across split-halves for each tract and method. 
- `calc_tract_wmh.py`: Calculates the Dice coefficient between tract segmentations and white matter hyperintensity (WMH) regions.

#### Fazekas Prediction Scripts

Fazekas score prediction scripts were used to train and save model predictions across multiple repeats and cross-validation folds.

- `prediction_multi-shell.ipynb`: Runs Fazekas score prediction for the multi-shell dataset for each method using a consistent data-splitting scheme. Saves the predicted values, probabilities, and true Fazekas scores as separate files.
- `prediction_single-shell.ipynb`: Runs Fazekas score prediction for the single-shell dataset for each method using the same data-splitting scheme. Saves the predicted values, probabilities, and true Fazekas scores as separate files.
- `prediction_helper.ipynb`: Compiles predicted, probabilties, and true Fazekas scores from model outputs across iterations.

---

### `3_figures/`

These scripts are numbered according to the figure number assigned in the manuscript. Letter subscripts (e.g., `figure01[a|b].ipynb`) generally refer to figure panel. Final manuscript figures were combined using [Inkscape](https://inkscape.org/).

- `figure01`: Low, medium, high reliability examples.
- `figure02`: fODF $r^{2}$ in normal appearing white matter (NAWM) and white matter hyperintensities (WMH).
- `figure03`: fODF glyph examples. 
- `figure04`: Tract weighted dice coefficient reliability.
- `figure05`: Profile ICC reliability.
- `figure06`: Tract yield.
- `figure07`: Tract and WMH weighted dice coefficient.
- `figure08`: Tract examples by Fazekas scores.
- `figure09`: Tract examples by tractography common errors.
- `figure10`: Tract profile examples by method.
- `figure11`: Fazkeas score receiver operating curves (ROC).

---

### `4_helper/`

- `create_glass_brain.[py|sh]`: Creates a ["glass brain"](https://neurosnippets.com/posts/glass-brain/) for visualization.
