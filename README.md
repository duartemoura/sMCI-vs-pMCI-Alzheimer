# Long-Term Prediction of Alzheimer's Disease Progression from MCI Using Combined 18F-FDG and Tau PET Imaging with Explainable Deep Learning

This repository contains the implementation of a hierarchical multi-stage 3D CNN framework for predicting progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) using PET neuroimaging data. The framework leverages cross-modality transfer learning to bridge the data availability gap between FDG-PET and Tau-PET imaging, and incorporates uncertainty quantification and explainability for clinical decision support.

## Overview

This project implements a deep learning pipeline that:

- **Stage 1**: Trains a foundation 3D CNN model on AD/CN classification using FDG-PET data
- **Stage 2**: Transfers learned representations to predict MCI-to-AD conversion using both FDG-PET and Tau-PET modalities
- **Uncertainty Estimation**: Uses Monte Carlo Dropout to provide confidence scores for predictions
- **Explainability**: Employs Grad-CAM to visualize disease-relevant brain regions

The methodology is described in detail in the associated Master's Thesis: *"Long-Term Prediction of Alzheimer's Disease Progression from MCI Using Combined 18F-FDG and Tau PET Imaging with Explainable Deep Learning"* (Universidad Carlos III de Madrid, 2026).

## Table of Contents

- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Project Structure](#project-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Usage](#usage)
- [Expected Outputs](#expected-outputs)
- [Key Features](#key-features)
- [References](#references)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Access to ADNI data (see [Data Requirements](#data-requirements))

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TFM_FDG
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The `requirements.txt` pins NumPy to version <2.0 for compatibility with compiled modules. If you encounter issues, ensure you're using NumPy 1.x.

4. **Install Jupyter (if not already installed):**
   ```bash
   pip install jupyter ipykernel
   ```

5. **Set up Jupyter kernel (optional, for local development):**
   ```bash
   python -m ipykernel install --user --name=tfm_fdg
   ```

### Key Dependencies

- **Deep Learning**: PyTorch, torchvision, torchaudio
- **Medical Imaging**: nibabel, SimpleITK, nilearn
- **Image Processing**: scikit-image, connected-components-3d
- **Hyperparameter Tuning**: optuna
- **Experiment Tracking**: wandb (optional but recommended)
- **Visualization**: matplotlib, seaborn, plotly

## Data Requirements

### ADNI Data Access

This project requires data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). You must:

1. Register for ADNI data access
2. Download the following:
   - **PET Imaging Data**: NIfTI format files for FDG-PET and/or Tau-PET (18F-flortaucipir/AV-1451)
   - **ADNIMERGE CSV**: Clinical and demographic data (`ADNIMERGE.csv`)
   - **Image Search Results**: CSV files with image metadata (e.g., `idaSearch_*.csv`)

### Data Organization

The pipeline expects the following structure:

```
TFM_FDG/
├── csvs/
│   ├── ADNIMERGE.csv              # Clinical/demographic data
│   └── idaSearch_*.csv             # Image metadata
├── <raw_data_directory>/          # Your ADNI NIfTI files
│   └── <subject_id>/
│       └── <scan_type>/
│           └── <date>/
│               └── *.nii.gz
└── ...
```

**Note:** Update the paths in each notebook to point to your actual data directories.

## Project Structure

```
TFM_FDG/
├── TAU/                           # Main pipeline notebooks
│   ├── 01_preprocess_and_organize.ipynb
│   ├── 02_preprocess_3d.ipynb
│   ├── 03_train_ad_cn_model.ipynb
│   ├── 04_mci_conversion_split.ipynb
│   ├── 05_create_kfold_datasets.ipynb
│   ├── 06_train_mci_conversion_model.ipynb
│   ├── 07_monte_carlo_dropout.ipynb
│   └── 08_analysis.ipynb
├── scripts/                       # Utility scripts
│   ├── analyze_ida_ad_cn.py
│   ├── analyze_ida_mci.py
│   ├── convert_dicom_to_nifti.py
│   └── count_tensor_files.py
├── data/
│   ├── processed/                 # Processed datasets (generated)
│   │   ├── tensors_fdg_last_run/
│   │   └── kfold__tau_final_run/
│   └── raw/                       # Raw data (user-provided)
├── csvs/                          # CSV metadata files
├── requirements.txt
└── README.md
```

## Pipeline Workflow

The pipeline consists of 8 sequential notebooks that must be executed in order:

### 1. `01_preprocess_and_organize.ipynb`
**Purpose**: Organize raw ADNI NIfTI files into structured folders based on diagnostic groups.

**Process**:
- Loads image metadata from CSV files (e.g., `idaSearch_*.csv`)
- Maps Image Data IDs to actual NIfTI files using subject ID, acquisition date, and scan description
- Organizes files into `ad/`, `cn/`, and `mci/` directories
- Handles multiple scans per subject (longitudinal data)
- Optionally filters MCI to baseline-only scans

**Inputs**:
- CSV file with image metadata (`Image Data ID`, `Subject`, `Group`, `Acq Date`, `Description`)
- Directory containing ADNI NIfTI files

**Outputs**:
- Organized directory structure:
  ```
  <output_path>/
  ├── ad/
  ├── cn/
  └── mci/
  ```

**Configuration**: Update paths for CSV file, base data directory, and output directory.

---

### 2. `02_preprocess_3d.ipynb`
**Purpose**: Preprocess 3D neuroimaging volumes for deep learning.

**Process**:
- **Brain Masking**: Applies Otsu thresholding + 3D connected component analysis to extract brain tissue
- **Resampling**: Resizes volumes to `100 × 100 × 90` voxels using trilinear interpolation
- **Normalization**: Per-volume min-max scaling to [0, 1] range
- **Data Splitting**: Subject-level split into train/validation/test sets (80/10/10)
- **Saving**: Serializes preprocessed data as PyTorch tensors in pickle files

**Inputs**:
- Organized AD and CN folders from Step 1

**Outputs**:
- `data/processed/tensors_fdg_last_run/`
  - `train/ad_cn_train.pkl`
  - `val/ad_cn_val.pkl`
  - `test/ad_cn_test.pkl`

Each pickle file contains a dictionary with:
- `images`: List of PyTorch tensors (shape: `[100, 100, 90]`)
- `labels`: List of binary labels (0=CN, 1=AD)
- `subject_ids`: List of subject identifiers

**Key Parameters**:
- Target shape: `(100, 100, 90)` voxels
- Normalization: Per-volume min-max scaling

---

### 3. `03_train_ad_cn_model.ipynb`
**Purpose**: Train the foundation 3D CNN model for AD/CN classification with hyperparameter optimization.

**Process**:
- **Architecture**: 3D CNN with convolutional blocks, batch normalization, dropout, and global average pooling
- **Hyperparameter Tuning**: Uses Optuna to optimize:
  - Learning rate
  - Dropout rate
  - Number of filters per layer
  - Optimizer choice (Adam/AdamW)
- **Data Augmentation**: Random rotations and translations during training
- **Training**: Trains on AD/CN data with validation monitoring
- **Experiment Tracking**: Logs to Weights & Biases (W&B) for visualization

**Inputs**:
- `ad_cn_train.pkl`, `ad_cn_val.pkl` from Step 2

**Outputs**:
- `ad_cn_model_best_tuned.pth`: Best model checkpoint
- `hyperparameter_study.pkl`: Optuna study results
- W&B artifacts (if enabled)
- Augmentation visualization examples

**Key Features**:
- Transfer learning foundation for MCI conversion models
- Systematic hyperparameter search
- Early stopping based on validation performance

---

### 4. `04_mci_conversion_split.ipynb`
**Purpose**: Classify MCI subjects into progressive MCI (pMCI) and stable MCI (sMCI) based on longitudinal follow-up.

**Process**:
- Loads ADNIMERGE CSV to track diagnostic trajectories
- **pMCI Criteria**: Subjects diagnosed as MCI/EMCI/LMCI at baseline who convert to Dementia within 24 months
- **sMCI Criteria**: Subjects who remain MCI for at least 24 months without conversion
- **Exclusions**: Subjects who revert to CN or lack sufficient follow-up
- Selects earliest scan per subject (by Image Data ID) to avoid temporal confounds
- Organizes files into `pMCI/` and `sMCI/` folders

**Inputs**:
- MCI folder from Step 1
- `ADNIMERGE.csv` with longitudinal diagnostic data

**Outputs**:
- `mci_conversion_split/`
  - `pMCI/`
  - `sMCI/`

**Key Parameters**:
- Conversion window: 24 months
- Minimum stability period: 24 months for sMCI

---

### 5. `05_create_kfold_datasets.ipynb`
**Purpose**: Create 5-fold cross-validation datasets for robust model evaluation.

**Process**:
- Shuffles pMCI and sMCI file lists
- Creates 5 stratified folds (subject-level splitting to prevent data leakage)
- For each fold:
  - Designates one fold as validation set
  - Combines remaining 4 folds as training set
  - Applies same preprocessing pipeline as Step 2 (masking, resampling, normalization)
- Saves processed data for each fold

**Inputs**:
- `pMCI/` and `sMCI/` folders from Step 4

**Outputs**:
- `data/processed/kfold__tau_final_run/`
  - `train_fold_1.pkl` through `train_fold_5.pkl`
  - `val_fold_1.pkl` through `val_fold_5.pkl`
  - `kfold_info.pkl`: Metadata about fold composition

**Key Features**:
- Subject-level splitting ensures no data leakage
- Stratified folds maintain class balance
- Same preprocessing ensures consistency with foundation model

---

### 6. `06_train_mci_conversion_model.ipynb`
**Purpose**: Train MCI-to-AD conversion prediction models using transfer learning.

**Process**:
- **Transfer Learning**: Loads pre-trained AD/CN model from Step 3
- **Fine-tuning**: Replaces final classification layer and fine-tunes on MCI conversion data
- **5-Fold Training**: Trains a separate model for each fold
- **Lower Learning Rate**: Uses reduced learning rate (e.g., 0.0001) for fine-tuning
- **Evaluation**: Tracks validation AUC and accuracy for each fold

**Inputs**:
- Pre-trained model: `ad_cn_model_best_tuned.pth` from Step 3
- K-fold datasets: `train_fold_*.pkl`, `val_fold_*.pkl` from Step 5

**Outputs**:
- `mci_conversion_models/`
  - `fold_1_best.pth` through `fold_5_best.pth`
- Training logs and metrics for each fold

**Key Features**:
- Leverages neurodegenerative representations learned from AD/CN classification
- Enables robust evaluation through cross-validation
- Supports both FDG-PET and Tau-PET modalities (with cross-modality transfer)

---

### 7. `07_monte_carlo_dropout.ipynb`
**Purpose**: Estimate prediction uncertainty using Monte Carlo Dropout.

**Process**:
- **MCDropout Layer**: Replaces standard dropout with custom layer that remains active during inference
- **Multiple Forward Passes**: Performs 500 forward passes per subject with dropout enabled
- **Uncertainty Calculation**: Computes mean prediction and standard deviation (confidence score)
- **Aggregation**: Collects results across all folds and subjects

**Inputs**:
- Trained models: `fold_*_best.pth` from Step 6
- Validation datasets: `val_fold_*.pkl` from Step 5

**Outputs**:
- `monte_carlo_results.pkl`: DataFrame with columns:
  - `subject_id`: Subject identifier
  - `fold`: Cross-validation fold
  - `label`: Ground truth (0=sMCI, 1=pMCI)
  - `mc_mean`: Mean prediction across 500 iterations
  - `mc_std`: Standard deviation (uncertainty/confidence score)
- Visualization plots of prediction distributions

**Key Features**:
- Enables reliability-aware risk stratification
- High-confidence predictions achieve ~90% accuracy (FDG-PET) and ~80% (Tau-PET)
- Identifies ambiguous cases for clinical review

---

### 8. `08_analysis.ipynb`
**Purpose**: Comprehensive analysis and visualization of model performance.

**Process**:
- **Performance Metrics**:
  - ROC-AUC, accuracy, sensitivity, specificity
  - Optimal threshold selection using G-Mean
  - Precision-Recall curves, confusion matrices
- **Correlation Analysis**:
  - Relationships between predictions and clinical variables (MMSE, MoCA, APOE4)
  - Uncertainty correlation with time-to-conversion
- **Time-to-Conversion Analysis**:
  - Performance across different conversion windows (<2 years, 2-4 years, >4 years)
  - Temporal sensitivity of FDG-PET vs. Tau-PET
- **Visualization**:
  - ROC curves, PR curves, scatter plots, correlation heatmaps
  - Grad-CAM visualizations (if implemented)

**Inputs**:
- `monte_carlo_results.pkl` from Step 7
- `ADNIMERGE.csv` for demographic/clinical data

**Outputs**:
- Comprehensive analysis report
- Performance metrics tables
- Visualization figures (saved to `reports/figures/`)

**Key Features**:
- Validates model predictions against established biomarkers
- Provides insights into temporal sensitivity
- Supports clinical interpretation

## Usage

### Running the Pipeline

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Execute notebooks sequentially:**
   - Open `TAU/01_preprocess_and_organize.ipynb`
   - Update paths in the first code cell to match your data locations
   - Run all cells
   - Repeat for each subsequent notebook in order

3. **Path Configuration:**
   Each notebook contains a "Define Paths" section at the beginning. Update these to match your environment:
   ```python
   # Example from notebook 01
   csv_file = Path("/path/to/your/idaSearch.csv")
   data_base_path = Path("/path/to/your/ADNI_NIFTI")
   output_path = Path("/path/to/output")
   ```

### Google Colab Usage

Some notebooks include Google Colab integration (e.g., `drive.mount()`). To use locally:
- Remove or comment out Colab-specific cells
- Update paths to local filesystem paths
- Ensure GPU access if training models

### Expected Runtime

- **Steps 1-2**: Data organization and preprocessing (~hours, depends on dataset size)
- **Step 3**: AD/CN model training (~hours to days, depends on hyperparameter search)
- **Steps 4-5**: MCI splitting and k-fold creation (~hours)
- **Step 6**: MCI conversion training (~hours per fold, 5 folds total)
- **Step 7**: Monte Carlo Dropout (~hours, 500 iterations per subject)
- **Step 8**: Analysis (~minutes)

**Total**: Several days to weeks depending on hardware and dataset size.

## Expected Outputs

### Model Checkpoints
- `ad_cn_model_best_tuned.pth`: Foundation AD/CN classification model
- `mci_conversion_models/fold_*_best.pth`: 5 MCI conversion models (one per fold)

### Processed Data
- `data/processed/tensors_fdg_last_run/`: Preprocessed AD/CN tensors
- `data/processed/kfold__tau_final_run/`: K-fold MCI conversion datasets

### Results
- `monte_carlo_results.pkl`: Uncertainty estimates for all subjects
- Analysis reports and visualizations

## Key Features

### Cross-Modality Transfer Learning
- Transfers representations from FDG-PET (large dataset) to Tau-PET (small dataset)
- Enables robust Tau-PET models despite 3x reduction in training data
- Maintains comparable classification accuracy

### Uncertainty Quantification
- Monte Carlo Dropout provides confidence scores
- Enables reliability-aware risk stratification
- High-confidence predictions achieve ~90% accuracy (FDG-PET)

### Explainability
- Grad-CAM visualizations highlight disease-relevant brain regions
- FDG-PET focuses on posterior cingulate and precuneus
- Tau-PET emphasizes medial temporal lobe (Braak stages I-II)

### Clinical Validation
- Correlates predictions with established biomarkers (MMSE, MoCA, APOE4)
- Temporal sensitivity analysis across conversion windows
- Biological interpretability of highlighted regions

## References

### Primary Reference
- **Master's Thesis**: "Long-Term Prediction of Alzheimer's Disease Progression from MCI Using Combined 18F-FDG and Tau PET Imaging with Explainable Deep Learning" (2026)
  - Duarte Pinto Correia de Moura
  - Universidad Carlos III de Madrid
  - Supervisor: David Izquierdo García

### Key Methodological References
- **ADNI**: Alzheimer's Disease Neuroimaging Initiative - [adni.loni.usc.edu](http://adni.loni.usc.edu/)
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017
- **Monte Carlo Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation," ICML 2016
- **Transfer Learning**: Fernandez-Garcia et al., "Improving confidence in long-term deep learning prediction of progression from MCI to AD using 18F-FDG-PET," Master's Thesis, UC3M 2024

### Data Citation
Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI).


## Acknowledgments

- ADNI for providing the neuroimaging and clinical data
- Universidad Carlos III de Madrid
- Supervisor: Prof. David Izquierdo García

## Contact

For questions or issues, please open an issue in the repository or contact the authors.

---

**Note**: This pipeline requires access to ADNI data. Ensure you have proper data use agreements in place before running the notebooks.
