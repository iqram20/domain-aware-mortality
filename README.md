# Domain-Aware Mortality Prediction

A **domain-aware, interpretable deep learning framework** for predicting perioperative mortality from routinely collected electronic health record (EHR) data.

This repository contains the code used to preprocess perioperative data, construct clinically meaningful feature domains, train and evaluate the **Domain-Aware Residual Network (DARN)**, and perform model interpretability and Responsible AI analyses.

---

## Overview

Perioperative mortality is a rare but clinically important outcome influenced by multiple interacting factors, including patient characteristics, comorbidities, preoperative laboratory measurements, surgical procedures, medications, and intraoperative physiology.

The framework in this repository organizes heterogeneous perioperative variables into clinically meaningful domains rather than treating all predictors as a single unstructured feature vector.

Each domain is processed through a dedicated representation-learning module, allowing the model to preserve clinical structure while learning nonlinear relationships across perioperative data.

The analysis pipeline additionally incorporates:

* Model discrimination and calibration
* Independent holdout evaluation
* Domain-aware representation learning
* Feature-level explainability
* Domain-level interpretability
* Subgroup performance assessment
* Responsible AI evaluation
* Reproducible model training and artifact generation

---

## DARN Framework

The **Domain-Aware Residual Network (DARN)** organizes perioperative predictors into clinically meaningful domains such as:

* Demographics
* Diagnoses and comorbidities
* Preoperative laboratory measurements
* Intraoperative vital signs
* Medications
* Procedures
* Surgical and anesthesia duration
* Clinically motivated interaction features

Domain-specific neural encoders learn representations independently before the information is integrated for mortality prediction.

This architecture is designed to preserve the clinical organization of the input data while allowing nonlinear relationships to be learned across heterogeneous perioperative information.

---

## Model Workflow

```text
Perioperative EHR Data
        │
        ▼
Data Cleaning and Cohort Construction
        │
        ▼
Preprocessing
 ├── Missing-value imputation
 ├── Continuous-variable scaling
 ├── Categorical encoding
 └── Variance filtering
        │
        ▼
Clinical Domain Construction
        │
        ├── Demographics
        ├── Diagnoses
        ├── Preoperative Labs
        ├── Intraoperative Vitals
        ├── Medications
        ├── Procedures
        ├── Durations
        └── Interaction Features
        │
        ▼
Domain-Specific DARN Encoders
        │
        ▼
Domain Fusion
        │
        ▼
Mortality Risk Prediction
        │
        ├── Discrimination
        ├── Calibration
        ├── Explainability
        └── Responsible AI Evaluation
```

---

## Responsible AI and Interpretability

The pipeline evaluates the model beyond overall predictive performance.

### Feature-Level Explainability

Model explanations identify individual perioperative variables contributing most strongly to predicted mortality risk.

The repository supports attribution-based analysis to characterize both the magnitude and direction of individual feature contributions.

### Domain-Level Interpretability

Feature-level explanations are aggregated into predefined clinical domains to determine how broader components of perioperative care contribute to model predictions.

Domain-level analyses provide an interpretation layer that more closely reflects clinical reasoning than isolated feature rankings.

### Domain Ablation

Clinical domains can be systematically removed from the model to quantify their contribution to predictive performance.

This provides a complementary evaluation of domain importance beyond attribution scores alone.

### Subgroup Evaluation

Model performance is evaluated across clinically relevant patient groups, including demographic and perioperative risk strata.

These analyses are intended to identify potential disparities in predictive performance and support responsible evaluation before clinical translation.

### Calibration

Predicted mortality probabilities are evaluated against observed event rates to assess whether estimated risks correspond to actual outcome frequencies.

---

## Data

This project uses the **INSPIRE perioperative dataset**, a publicly available and de-identified perioperative EHR resource derived from Seoul National University Hospital.

INSPIRE is available through PhysioNet:

https://physionet.org/content/inspire/

The original INSPIRE data are **not redistributed in this repository**.

Users should obtain the dataset directly from PhysioNet and comply with the corresponding data-use requirements.

---

## Repository Contents

The repository contains code for the major stages of the modeling pipeline, including:

```text
domain-aware-mortality/
│
├── README.md
│
├── notebooks/
│   └── analysis notebooks
│
├── src/
│   ├── preprocessing
│   ├── feature engineering
│   ├── domain construction
│   ├── model definition
│   ├── training
│   ├── evaluation
│   └── explainability
│
├── figures/
│   └── manuscript and analysis figures
│
├── results/
│   └── generated model outputs
│
└── requirements.txt
```

The exact directory structure may vary depending on the released version of the analysis code.

---

## Main Analysis Components

The workflow includes:

1. Cohort construction
2. Outcome definition
3. Perioperative feature extraction
4. Missing-data preprocessing
5. Categorical one-hot encoding
6. Continuous-variable scaling
7. Variance-based feature filtering
8. Clinical domain assignment
9. Training/development and independent holdout separation
10. DARN model training
11. Baseline-model comparison
12. Performance evaluation
13. Calibration analysis
14. Feature attribution
15. Domain-level interpretation
16. Domain ablation
17. Subgroup and Responsible AI analyses
18. Figure and table generation

All preprocessing transformations used for model development should be fitted using the training data only and subsequently applied unchanged to validation or holdout data to prevent information leakage.

---

## Evaluation Metrics

Mortality prediction is evaluated using complementary discrimination and calibration metrics, including:

* Area under the receiver operating characteristic curve (**AUROC**)
* Area under the precision-recall curve (**AUPRC**)
* Brier score
* Sensitivity
* Specificity
* Positive predictive value
* Negative predictive value
* Calibration characteristics
* Risk concentration across high-risk prediction strata

For a rare outcome such as perioperative mortality, AUPRC and event prevalence should be considered alongside AUROC when interpreting model performance.

---

## Example Figure Integration

Manuscript figures can be displayed directly in the GitHub README.

For example:

```markdown
## Model Framework

<p align="center">
  <img src="figures/Figure1.png" width="900">
</p>

**Figure 1.** Domain-aware framework for perioperative mortality prediction.
```

Additional figures can be added in the same way:

```markdown
## Model Performance

<p align="center">
  <img src="figures/Figure_Performance.png" width="900">
</p>
```

Recommended organization:

```text
figures/
├── Figure1_Framework.png
├── Figure2_Cohort.png
├── Figure3_Performance.png
├── Figure4_Explainability.png
├── Figure5_Domain_Analysis.png
└── Figure6_Responsible_AI.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/iqram20/domain-aware-mortality.git
cd domain-aware-mortality
```

Create a Python environment:

```bash
conda create -n darn-mortality python=3.10
conda activate darn-mortality
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Analysis

After obtaining and preparing the INSPIRE dataset, update the appropriate input and output paths in the analysis configuration or notebook.

The general execution order is:

```text
1. Load INSPIRE data
2. Construct the study cohort
3. Generate perioperative predictors
4. Preprocess and encode variables
5. Assign features to clinical domains
6. Create development and holdout datasets
7. Train DARN
8. Evaluate the locked holdout cohort
9. Generate model explanations
10. Perform subgroup and Responsible AI analyses
11. Export figures, tables, and model artifacts
```

Because the original clinical dataset is not distributed with this repository, local paths must be configured before reproducing the complete analysis.

---

## Reproducibility

To improve reproducibility, the analysis uses fixed random seeds for data splitting and model initialization where applicable.

Important model artifacts should include:

* Train/validation/test assignments
* Selected feature lists
* Clinical domain mappings
* Preprocessing parameters
* Trained model checkpoints
* Predicted probabilities
* Evaluation metrics
* Calibration outputs
* Feature attribution matrices
* Domain-level attribution results
* Subgroup evaluation results

Researchers reproducing the analysis should preserve subject-level separation between model-development and holdout datasets.

---

## Intended Use

This framework is intended for **research purposes** and methodological evaluation of perioperative risk prediction.

The model is **not intended for direct clinical use without external validation**.

Performance should be independently assessed in geographically and clinically distinct populations before considering clinical implementation. Differences in patient populations, surgical practice, coding systems, measurement frequency, missing-data patterns, and institutional workflows may affect model transportability.

---

## Code Availability

The implementation of the domain-aware mortality prediction framework is available in this repository:

https://github.com/iqram20/domain-aware-mortality

---

## Data Availability

The study uses the publicly available INSPIRE perioperative dataset.

**INSPIRE:**
https://physionet.org/content/inspire/

No patient-level INSPIRE data are redistributed through this repository.

---

## Citation

If you use this code or framework in your research, please cite the associated manuscript.

```bibtex
@article{hussain_darn_mortality,
  title   = {Domain-Aware Interpretable Deep Learning for Perioperative Mortality Prediction},
  author  = {Hussain, Iqram and collaborators},
  journal = {To be updated},
  year    = {2026}
}
```

The citation information should be updated after final publication.

---

## Authors

**Iqram Hussain, PhD**
Department of Anesthesiology
Weill Cornell Medicine
Cornell University
New York, NY, USA



---

## License

Please refer to the repository license for conditions governing reuse of the source code.

The INSPIRE dataset remains subject to the terms and conditions established by PhysioNet and the original dataset providers.

---

## Disclaimer

This repository is intended for research and educational use. Predictions generated by the models should not be interpreted as medical advice or used independently for clinical decision-making without appropriate external validation, regulatory review, and prospective clinical evaluation.
