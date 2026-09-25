# ============================================================================
# IMPROVED UNIFORM DARN + DARN-XGBOOST HYBRID MORTALITY PIPELINE
# ============================================================================
# Models:
#   1. Uniform DARN (5-fold development ensemble)
#   2. ASA-only logistic regression (clinical baseline)
#   3. Multivariable logistic regression
#   4. XGBoost (when xgboost is installed)
#   5. Standard multilayer perceptron (MLP)
#   6. Hybrid DARN-XGBoost probability blend (OOF weight selection)
#   7. Hybrid DARN-XGBoost cross-fitted logistic stack
#
# Reviewer-response additions:
#   - Excludes ASA 6 organ-donor encounters before splitting.
#   - Adds ASA-only clinical baseline, two prespecified operating points,
#     decision-curve analysis, risk concentration, subgroup fairness CIs,
#     equalized-odds gaps, error analysis, calibration confidence bands,
#     Hosmer-Lemeshow diagnostics, variance-filter audit, direct DARN
#     GradientSHAP, XGBoost contributions, input-domain masking,
#     true leave-one-domain-out retraining, paired bootstrap performance
#     differences, false-negative case review, clinically focused tail
#     calibration, and targeted eye/ear diagnosis sensitivity analyses.
#
# Methodological safeguards:
#   - Preserves a fixed, untouched 20% outer test cohort.
#   - Uses stratified cross-validation only inside the 80% development cohort.
#   - Fits preprocessing, variance filtering, and supervised feature selection
#     separately inside each development fold.
#   - Creates pooled out-of-fold probabilities for threshold selection and
#     Platt calibration; test labels are never used for model selection.
#   - Evaluates every model on the exact same outer test encounters.
#   - Computes stratified bootstrap confidence intervals and paired differences.
#
# Uniform DARN improvements:
#   - Smaller residual domain encoders for rare-event data.
#   - Stronger dropout and L2 regularization.
#   - Input feature dropout.
#   - Symmetric equal-domain fusion using mean, standard deviation, maximum,
#     and root-mean-square statistics. No learned domain weighting is used.
#   - Moderated positive weighting and mild focal loss.
#   - Training-only domain-balanced feature selection.
#
# Hybrid safeguards:
#   - Hybrid parameters are learned only from development out-of-fold predictions.
#   - The outer test labels are never used to select blend weights or stacker coefficients.
#   - The hybrid is reported as a distinct DARN-XGBoost model, not as pure DARN.
#
# Suggested cluster execution:
#   python -u Mortality_DARN_full_pipe_v6_reviewer_complete.py
#
# Computational note:
#   True leave-one-domain-out retraining refits preprocessing, feature
#   selection, five Uniform DARN folds, and—unless disabled—five XGBoost
#   folds for every removed domain. This is intentionally more expensive
#   than test-time input masking.
# ============================================================================

from __future__ import annotations

import argparse
import copy
import datetime
import inspect
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"

try:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False


try:
    import statsmodels.api as sm

    STATSMODELS_AVAILABLE = True
except Exception:
    sm = None
    STATSMODELS_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class PipelineConfig:
    data_path: str = (
        "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/"
        "Results/Final_Datasets/df_final_mortality_2026-05-24_12-44_temporal_safe.csv"
    )
    out_root: str = "darn_cv_hybrid_runs"
    target_column: str = "mortality_30d"
    test_size: float = 0.20
    random_state: int = 42
    n_folds: int = 5
    variance_threshold: float = 1e-4
    add_interactions: bool = True
    exclude_asa6: bool = True

    # Primary reviewer-facing model.
    primary_model_name: str = "Hybrid DARN-XGB Blend"

    # Reviewer-facing cohort and descriptive analyses
    correlation_top_n: int = 30

    # Uniform DARN
    darn_seeds_per_fold: int = 1
    darn_hidden_dim: int = 48
    darn_dropout: float = 0.30
    darn_feature_dropout: float = 0.10
    darn_learning_rate: float = 3e-4
    darn_weight_decay: float = 1e-4
    darn_batch_size: int = 1024
    prediction_batch_size: int = 4096
    darn_max_epochs: int = 120
    darn_patience: int = 12
    darn_focal_gamma: float = 0.5
    darn_pos_weight: float = 10.0

    # MLP baseline
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.30
    mlp_learning_rate: float = 3e-4
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 1024
    mlp_max_epochs: int = 100
    mlp_patience: int = 12
    mlp_pos_weight: float = 10.0

    # Logistic regression
    logistic_c: float = 0.10
    logistic_max_iter: int = 3000

    # XGBoost
    run_xgboost: bool = True
    xgb_n_estimators: int = 2000
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 3
    xgb_min_child_weight: float = 5.0
    xgb_subsample: float = 0.80
    xgb_colsample_bytree: float = 0.80
    xgb_reg_alpha: float = 0.10
    xgb_reg_lambda: float = 2.0
    xgb_early_stopping_rounds: int = 75
    xgb_n_jobs: int = 8

    # Leakage-safe DARN-XGBoost hybrids. These require XGBoost.
    run_hybrid: bool = True
    hybrid_blend_grid_size: int = 181
    hybrid_min_darn_weight: float = 0.10
    hybrid_max_darn_weight: float = 0.90
    hybrid_stack_folds: int = 5
    hybrid_stack_c: float = 0.50
    hybrid_stack_class_weight_balanced: bool = True

    # Operating point and evaluation
    threshold_strategy: str = "sensitivity"
    target_sensitivity: float = 0.80
    selective_target_specificity: float = 0.95

    # Calibration and threshold transport
    # Each base learner is calibrated with a cross-fitted Platt model inside
    # its held-out development fold. The fold-specific calibrator is then fit
    # on the complete held-out fold and applied to that fold model's test
    # predictions before the five test predictions are averaged.
    platt_c: float = 1.0
    platt_crossfit_folds: int = 3
    # Absolute transport is the cleanest reviewer-facing default after
    # leakage-safe calibration. Alert-rate transport remains available as an
    # explicit sensitivity option but is not used by default.
    threshold_transport_mode: str = "absolute"  # absolute, alert_rate, or auto
    threshold_transport_rate_ratio: float = 3.0
    threshold_transport_absolute_gap: float = 0.25
    threshold_transport_degenerate_rate: float = 0.99

    n_bootstrap: int = 2000
    subgroup_bootstrap: int = 500
    calibration_bootstrap: int = 500
    calibration_bins: int = 10
    subgroup_min_events: int = 3
    subgroup_min_nonevents: int = 20
    decision_curve_min_threshold: float = 0.0005
    decision_curve_max_threshold: float = 0.05
    decision_curve_points: int = 100

    # Training-only per-domain feature caps
    cap_demographics: int = 20
    cap_diagnoses: int = 40
    cap_comorbidity: int = 40
    cap_durations: int = 10
    cap_interactions: int = 20
    cap_intraop_vitals: int = 180
    cap_medications: int = 120
    cap_preop_labs: int = 100
    cap_procedures: int = 200
    cap_other: int = 50

    # Optional explainability and reviewer sensitivity analyses.
    run_gradient_shap: bool = True
    run_xgb_contributions: bool = True
    max_shap_n: int = 1000
    shap_gradient_samples: int = 12

    # Existing inexpensive perturbation analysis: the fitted DARN is retained
    # and one transformed input domain is set to zero at test time.
    run_input_domain_masking: bool = True

    # True domain-removal analysis. Preprocessing, variance filtering,
    # supervised feature selection, calibration, thresholds, DARN, XGBoost,
    # and the probability blend are all rebuilt from development data.
    run_leave_one_domain_out_retraining: bool = True
    lodo_retrain_hybrid: bool = True
    lodo_bootstrap: int = 2000
    lodo_save_models: bool = False

    # Targeted sensitivity analysis for the unexpected eye/ear diagnosis
    # feature. The adjusted association analysis is inexpensive; the optional
    # feature-removal retraining adds five DARN and five XGBoost fits.
    run_eye_ear_adjusted_analysis: bool = True
    run_eye_ear_feature_retraining: bool = True

    # Reviewer-facing rare-event calibration and error review.
    run_tail_risk_strata_calibration: bool = True
    run_false_negative_review: bool = True

    def domain_caps(self) -> Dict[str, int]:
        return {
            "Demographics": self.cap_demographics,
            "Diagnoses": self.cap_diagnoses,
            "Comorbidity": self.cap_comorbidity,
            "Durations": self.cap_durations,
            "Interactions": self.cap_interactions,
            "Intraop_Vitals": self.cap_intraop_vitals,
            "Medications": self.cap_medications,
            "Preop_Labs": self.cap_preop_labs,
            "Procedures": self.cap_procedures,
            "Other": self.cap_other,
        }


# ============================================================================
# REPRODUCIBILITY AND UTILITIES
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Cannot serialize object of type {type(value)}")


def safe_logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def make_run_directory(out_root: str) -> pathlib.Path:
    root = pathlib.Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "fold_artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir



# ============================================================================
# COHORT AUDIT AND REVIEWER-SPECIFIC EXCLUSIONS
# ============================================================================


def parse_asa_numeric(values: pd.Series) -> pd.Series:
    """Parse numeric ASA class from numeric or text representations."""
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        extracted = (
            values.astype("string")
            .str.extract(r"(?i)(?:ASA\s*)?([1-6])", expand=False)
        )
        numeric.loc[missing] = pd.to_numeric(extracted.loc[missing], errors="coerce")
    return numeric.astype(float)


def apply_reviewer_cohort_exclusions(
    df: pd.DataFrame,
    target_column: str,
    exclude_asa6: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply auditable cohort exclusions before any splitting or modeling."""
    working = df.copy()
    working["_source_row_position"] = np.arange(len(working), dtype=int)
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "step": "Raw input rows",
            "N": int(len(working)),
            "Events": int(pd.to_numeric(working.get(target_column), errors="coerce").fillna(0).sum()),
            "Excluded_at_step": 0,
        }
    )

    target_mask = working[target_column].notna()
    excluded_missing_target = int((~target_mask).sum())
    working = working.loc[target_mask].copy()
    rows.append(
        {
            "step": "Nonmissing mortality outcome",
            "N": int(len(working)),
            "Events": int(pd.to_numeric(working[target_column], errors="coerce").fillna(0).sum()),
            "Excluded_at_step": excluded_missing_target,
        }
    )

    if exclude_asa6 and "asa" in working.columns:
        asa_numeric = parse_asa_numeric(working["asa"])
        asa6_mask = asa_numeric.eq(6)
        excluded_asa6 = int(asa6_mask.sum())
        excluded_asa6_events = int(
            pd.to_numeric(working.loc[asa6_mask, target_column], errors="coerce")
            .fillna(0)
            .sum()
        )
        working = working.loc[~asa6_mask].copy()
        rows.append(
            {
                "step": "Exclude ASA 6 organ-donor encounters",
                "N": int(len(working)),
                "Events": int(pd.to_numeric(working[target_column], errors="coerce").fillna(0).sum()),
                "Excluded_at_step": excluded_asa6,
                "Excluded_events_at_step": excluded_asa6_events,
            }
        )
    else:
        rows.append(
            {
                "step": "ASA 6 exclusion not applied",
                "N": int(len(working)),
                "Events": int(pd.to_numeric(working[target_column], errors="coerce").fillna(0).sum()),
                "Excluded_at_step": 0,
            }
        )

    return working.reset_index(drop=True), pd.DataFrame(rows)


def cohort_audit_summary(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Create a compact audit table resolving cohort and emergency-case reporting."""
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    rows: List[Dict[str, Any]] = [
        {
            "characteristic": "Overall cohort",
            "level": "All",
            "N": int(len(df)),
            "Deaths": int(y.sum()),
            "Mortality_rate": float(y.mean()),
        }
    ]

    for column, label in (("emop", "Emergency surgery"), ("asa", "ASA class"), ("sex", "Sex")):
        if column not in df.columns:
            continue
        if column == "asa":
            values = parse_asa_numeric(df[column]).astype("Int64").astype("string")
        elif column == "emop":
            numeric = pd.to_numeric(df[column], errors="coerce")
            values = numeric.map({0.0: "Elective", 1.0: "Emergency"}).fillna(df[column].astype("string"))
        else:
            values = df[column].astype("string").fillna("Missing")

        # Pandas can raise ``Categorical categories cannot be null`` when
        # grouping nullable String/Int64 values with ``dropna=False``. Build a
        # plain object/string audit frame with an explicit Missing sentinel,
        # then aggregate outcomes by level. This is also index-safe because it
        # uses aligned NumPy arrays rather than treating DataFrame labels as
        # positional indices.
        audit_levels = values.astype("object")
        audit_levels = audit_levels.where(pd.notna(audit_levels), "Missing")
        audit_levels = audit_levels.astype(str)

        audit_frame = pd.DataFrame(
            {
                "level": audit_levels.to_numpy(),
                "outcome": y.to_numpy(dtype=int),
            }
        )
        grouped = (
            audit_frame.groupby("level", sort=True, observed=False)["outcome"]
            .agg(N="size", Deaths="sum")
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            n_level = int(row.N)
            deaths_level = int(row.Deaths)
            rows.append(
                {
                    "characteristic": label,
                    "level": str(row.level),
                    "N": n_level,
                    "Deaths": deaths_level,
                    "Mortality_rate": (
                        float(deaths_level / n_level) if n_level else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# FEATURE ENGINEERING AND EXTRACTION
# ============================================================================


def _numeric_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def add_mortality_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small prespecified set of outcome-independent interactions."""

    df = df.copy()
    emergency = _numeric_series(df, "emop")
    age = _numeric_series(df, "age")
    asa = _numeric_series(df, "asa")

    for column in ("or_duration", "anesthesia_duration", "surgery_duration"):
        duration = _numeric_series(df, column)
        if duration is None:
            continue

        log_duration = np.log1p(duration.clip(lower=0))

        if emergency is not None:
            df[f"INT_emergency_x_{column}"] = emergency * log_duration
        if age is not None:
            df[f"INT_age_x_{column}"] = age * log_duration
        if asa is not None:
            df[f"INT_asa_x_{column}"] = asa * log_duration

    if age is not None and emergency is not None:
        df["INT_age_x_emergency"] = age * emergency

    if asa is not None and emergency is not None:
        df["INT_asa_x_emergency"] = asa * emergency

    return df


def extract_model_features_classification(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    drop_columns = {
        "mortality_30d",
        "LOS_days",
        "hospital_los",
        "postop_hospital_los",
        "postop_hospital_los_days",
        "Died_in_hospital",
        "Died_30days",
        "inhosp_death_time",
        "allcause_death_time",
        "death_postop_time",
        "death_postdischarge",
        "discharge_time",
    }

    include_prefixes = (
        "preop_",
        "intraop_",
        "DIAG_",
        "CCI_",
        "PROC_",
        "preop_med_",
        "INT_",
    )

    static_variables = {
        "age",
        "sex",
        "race",
        "asa",
        "bmi",
        "emop",
        "surgery_duration",
        "or_duration",
        "anesthesia_duration",
    }

    feature_columns = [
        column
        for column in df.columns
        if (
            column.startswith(include_prefixes)
            or column in static_variables
        )
        and column not in drop_columns
    ]

    feature_columns = list(dict.fromkeys(feature_columns))
    X = df.loc[:, feature_columns].copy()

    if verbose:
        print(f"\nSelected raw predictors: {X.shape[1]:,}")

    return X


# ============================================================================
# DOMAIN ASSIGNMENT
# ============================================================================


def assign_raw_feature_domain(column: str) -> str:
    if column.startswith("INT_"):
        return "Interactions"
    if column.startswith("preop_med_"):
        return "Medications"
    if column.startswith("preop_"):
        return "Preop_Labs"
    if column.startswith("intraop_"):
        return "Intraop_Vitals"
    if column.startswith("DIAG_"):
        return "Diagnoses"
    if column.startswith("CCI_"):
        # Comorbidity indicators are treated as diagnosis-domain inputs in the
        # current eight-domain manuscript plan.
        return "Diagnoses"
    if column.startswith("PROC_"):
        return "Procedures"
    if column in {"surgery_duration", "or_duration", "anesthesia_duration"}:
        return "Durations"
    if column in {"age", "sex", "race", "asa", "bmi", "emop"}:
        return "Demographics"
    return "Other"


def get_feature_groups(raw_feature_columns: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for column in raw_feature_columns:
        domain = assign_raw_feature_domain(str(column))
        groups.setdefault(domain, []).append(str(column))
    return groups


def get_feature_groups_transformed(
    preprocessor: ColumnTransformer,
    feature_groups_raw: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    transformed_names = np.asarray(
        preprocessor.get_feature_names_out(), dtype=object
    )

    domain_output: Dict[str, List[str]] = {}

    for domain, raw_features in feature_groups_raw.items():
        matched: List[str] = []

        for raw_feature in raw_features:
            numeric_name = f"num__{raw_feature}"
            missing_name = f"num__missingindicator_{raw_feature}"
            categorical_prefix = f"cat__{raw_feature}_"

            for transformed_name in transformed_names:
                transformed_name = str(transformed_name)
                if (
                    transformed_name == numeric_name
                    or transformed_name == missing_name
                    or transformed_name.startswith(categorical_prefix)
                ):
                    matched.append(transformed_name)

        if matched:
            domain_output[domain] = list(dict.fromkeys(matched))

    return domain_output


# ============================================================================
# CLEANING AND PREPROCESSING
# ============================================================================


def clean_raw_predictors(
    X: pd.DataFrame,
    max_abs_value: float = 1e12,
) -> pd.DataFrame:
    """Convert booleans safely and normalize invalid values."""

    X = X.copy()

    boolean_columns = [
        column
        for column in X.columns
        if pd.api.types.is_bool_dtype(X[column].dtype)
    ]

    for column in boolean_columns:
        X[column] = (
            X[column]
            .map({False: 0.0, True: 1.0})
            .astype("float64")
        )

    X = X.replace({pd.NA: np.nan})
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_columns = X.select_dtypes(include=np.number).columns
    if len(numeric_columns) > 0:
        X.loc[:, numeric_columns] = X.loc[:, numeric_columns].clip(
            lower=-max_abs_value,
            upper=max_abs_value,
        )

    # Some CSV columns containing booleans plus missing values are loaded as
    # object dtype. OneHotEncoder rejects mixed bool/string object arrays, so
    # normalize every remaining nonnumeric value to a string while preserving
    # true missing values for the categorical imputer.
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        values = X[column]
        missing_mask = values.isna()
        normalized = values.astype(str)
        normalized.loc[missing_mask] = np.nan
        X[column] = normalized.astype("object")

    return X


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    X_train = clean_raw_predictors(X_train)

    numeric_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    ohe_parameters = inspect.signature(OneHotEncoder).parameters
    ohe_kwargs: Dict[str, Any] = {}

    if "sparse_output" in ohe_parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", **ohe_kwargs),
            ),
        ]
    )

    transformers: List[Tuple[str, Any, List[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No usable predictors were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_domain_indices(
    preprocessor: ColumnTransformer,
    variance_selector: VarianceThreshold,
    raw_feature_columns: Sequence[str],
) -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    all_transformed_names = np.asarray(
        preprocessor.get_feature_names_out(), dtype=object
    )

    retained_feature_names = all_transformed_names[
        variance_selector.get_support()
    ]

    raw_groups = get_feature_groups(raw_feature_columns)
    transformed_groups = get_feature_groups_transformed(preprocessor, raw_groups)

    name_to_index = {
        str(name): index for index, name in enumerate(retained_feature_names)
    }

    domain_indices: Dict[str, np.ndarray] = {}

    for domain, names in transformed_groups.items():
        indices = sorted(
            {
                name_to_index[name]
                for name in names
                if name in name_to_index
            }
        )
        if indices:
            domain_indices[domain] = np.asarray(indices, dtype=np.int64)

    assigned: List[int] = []
    for indices in domain_indices.values():
        assigned.extend(indices.tolist())

    duplicate_count = len(assigned) - len(set(assigned))
    if duplicate_count:
        raise ValueError(
            f"Feature-domain mapping has {duplicate_count} duplicates."
        )

    all_indices = set(range(len(retained_feature_names)))
    unassigned = sorted(all_indices - set(assigned))
    if unassigned:
        domain_indices["Other"] = np.asarray(unassigned, dtype=np.int64)

    domain_names = sorted(domain_indices)
    return domain_names, domain_indices, retained_feature_names


# ============================================================================
# TRAINING-ONLY DOMAIN-BALANCED FEATURE SELECTION
# ============================================================================


def standardized_mean_difference_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Fast supervised ranking fitted only on a training fold."""

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    positive = X[y == 1]
    negative = X[y == 0]

    if len(positive) == 0 or len(negative) == 0:
        return np.zeros(X.shape[1], dtype=float)

    mean_difference = np.nanmean(positive, axis=0) - np.nanmean(negative, axis=0)
    pooled_scale = np.nanstd(X, axis=0)
    scores = np.abs(mean_difference) / (pooled_scale + 1e-8)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def select_features_by_domain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    domain_names: Sequence[str],
    domain_indices: Dict[str, np.ndarray],
    retained_feature_names: np.ndarray,
    domain_caps: Dict[str, int],
) -> Tuple[np.ndarray, List[str], List[np.ndarray], Dict[str, np.ndarray], np.ndarray, pd.DataFrame, pd.DataFrame]:
    scores = standardized_mean_difference_scores(X_train, y_train)

    selected_original_indices: List[int] = []
    selection_rows: List[Dict[str, Any]] = []

    for domain in domain_names:
        indices = np.asarray(domain_indices[domain], dtype=np.int64)
        cap = int(domain_caps.get(domain, len(indices)))
        keep_count = min(len(indices), max(1, cap))

        local_order = np.argsort(scores[indices])[::-1]
        kept = indices[local_order[:keep_count]]
        selected_original_indices.extend(kept.tolist())

        selection_rows.append(
            {
                "domain": domain,
                "available_features": int(len(indices)),
                "selected_features": int(len(kept)),
                "cap": int(cap),
            }
        )

    # Preserve domain-grouped order for easier interpretation.
    selected_original_indices_array = np.asarray(
        selected_original_indices, dtype=np.int64
    )

    original_to_selected = {
        original_index: selected_index
        for selected_index, original_index in enumerate(
            selected_original_indices_array.tolist()
        )
    }

    selected_domain_indices: Dict[str, np.ndarray] = {}
    selected_domain_names: List[str] = []
    selected_domain_slices: List[np.ndarray] = []

    for domain in domain_names:
        mapped = [
            original_to_selected[index]
            for index in domain_indices[domain].tolist()
            if index in original_to_selected
        ]
        if mapped:
            mapped_array = np.asarray(mapped, dtype=np.int64)
            selected_domain_indices[domain] = mapped_array
            selected_domain_names.append(domain)
            selected_domain_slices.append(mapped_array)

    selected_feature_names = retained_feature_names[
        selected_original_indices_array
    ]

    feature_score_table = pd.DataFrame(
        {
            "feature": retained_feature_names,
            "selection_score": scores,
        }
    ).sort_values("selection_score", ascending=False)

    selection_summary = pd.DataFrame(selection_rows)

    return (
        selected_original_indices_array,
        selected_domain_names,
        selected_domain_slices,
        selected_domain_indices,
        selected_feature_names,
        selection_summary,
        feature_score_table,
    )


# ============================================================================
# PYTORCH DATASETS AND MODELS
# ============================================================================


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class FeatureDropout(nn.Module):
    def __init__(self, probability: float = 0.10):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return x

        keep_mask = torch.rand_like(x).ge(self.probability).to(x.dtype)
        return x * keep_mask / max(1e-8, 1.0 - self.probability)


class ResidualDomainEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        return self.output_norm(h + self.residual_block(h))


class UniformDARNClassifier(nn.Module):
    """Equal-domain fusion; no learned attention or domain preference."""

    def __init__(
        self,
        domain_slices: Sequence[np.ndarray],
        hidden_dim: int = 48,
        dropout: float = 0.30,
        feature_dropout: float = 0.10,
    ):
        super().__init__()

        if len(domain_slices) < 2:
            raise ValueError("Uniform DARN requires at least two domains.")

        self.n_domains = len(domain_slices)
        self.feature_dropout = FeatureDropout(feature_dropout)
        self.domain_encoders = nn.ModuleList()

        for domain_number, indices in enumerate(domain_slices):
            index_tensor = torch.as_tensor(indices, dtype=torch.long)
            self.register_buffer(f"domain_indices_{domain_number}", index_tensor)

            self.domain_encoders.append(
                ResidualDomainEncoder(
                    input_dim=len(indices),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # mean + standard deviation + maximum + root mean square
        fusion_dim = hidden_dim * 4

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        domain_vectors: List[torch.Tensor] = []

        for domain_number, encoder in enumerate(self.domain_encoders):
            indices = getattr(self, f"domain_indices_{domain_number}")
            domain_input = x.index_select(1, indices)
            domain_input = self.feature_dropout(domain_input)
            domain_vectors.append(encoder(domain_input))

        H = torch.stack(domain_vectors, dim=1)

        z_mean = H.mean(dim=1)
        z_std = H.std(dim=1, unbiased=False)
        z_max = H.max(dim=1).values
        z_rms = torch.sqrt(H.pow(2).mean(dim=1) + 1e-8)

        fused = torch.cat([z_mean, z_std, z_max, z_rms], dim=1)
        return self.head(fused).squeeze(-1)


class MLPBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()

        second_dim = max(32, hidden_dim // 2)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_dim),
            nn.LayerNorm(second_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(second_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float, gamma: float = 0.5):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        if self.gamma <= 0.0:
            return bce.mean()

        probabilities = torch.sigmoid(logits)
        pt = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        return (((1.0 - pt).pow(self.gamma)) * bce).mean()


# ============================================================================
# PYTORCH TRAINING AND PREDICTION
# ============================================================================


def predict_torch_model(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: List[np.ndarray] = []

    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    loader = DataLoader(
        X_tensor,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities)


def train_torch_binary_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight: float,
    focal_gamma: float,
    seed: int,
    label: str,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, float]]:
    """
    Strict-OOF training.

    X_validation / y_validation are the OUTER validation fold and are
    deliberately not used for early stopping or model selection.

    A stratified 20% split of X_train is used only for selecting the stopping
    epoch. The model is then reset to its original initialization and refit on
    all outer-training observations for the selected number of epochs.

    The learning-rate sequence observed during inner early stopping is replayed
    during the full outer-training refit.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    set_seed(seed)

    y_train = np.asarray(y_train, dtype=int)

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"{label}: outer-training data must contain both classes."
        )

    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    inner_train_indices, inner_validation_indices = next(
        inner_splitter.split(
            np.zeros(len(y_train)),
            y_train,
        )
    )

    X_inner_train = X_train[inner_train_indices]
    y_inner_train = y_train[inner_train_indices]

    X_inner_validation = X_train[inner_validation_indices]
    y_inner_validation = y_train[inner_validation_indices]

    print(
        f"{label} | STRICT OOF inner split | "
        f"train N={len(y_inner_train):,}, "
        f"events={int(y_inner_train.sum()):,} | "
        f"early-stop N={len(y_inner_validation):,}, "
        f"events={int(y_inner_validation.sum()):,}"
    )

    # Preserve the original randomly initialized model so that the final
    # full-outer-training refit can start from exactly the same initialization.
    initial_state = copy.deepcopy(model.state_dict())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    inner_dataset = TabularDataset(
        X_inner_train,
        y_inner_train,
    )
    inner_loader = DataLoader(
        inner_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = FocalBCEWithLogitsLoss(
        pos_weight=pos_weight,
        gamma=focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_ap = -np.inf
    best_auc = -np.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # INNER EARLY-STOPPING STAGE
    # ------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in inner_loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )
            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X_batch)
            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(X_batch)
            )

        train_loss = (
            running_loss
            / len(inner_dataset)
        )

        validation_probabilities = predict_torch_model(
            model,
            X_inner_validation,
            batch_size=max(
                batch_size,
                2048,
            ),
            device=device,
        )

        validation_auc = roc_auc_score(
            y_inner_validation,
            validation_probabilities,
        )

        validation_ap = average_precision_score(
            y_inner_validation,
            validation_probabilities,
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_AUROC": validation_auc,
                "validation_AUPRC": validation_ap,
                "learning_rate": current_lr,
                "phase": "inner_early_stopping",
            }
        )

        print(
            f"{label} | Inner epoch {epoch:03d} | "
            f"Loss={train_loss:.6f} | "
            f"Inner AUROC={validation_auc:.4f} | "
            f"Inner AUPRC={validation_ap:.4f} | "
            f"LR={current_lr:.2e}"
        )

        scheduler.step(
            validation_ap
        )

        improved = (
            validation_ap > best_ap + 1e-5
            or (
                abs(
                    validation_ap - best_ap
                ) <= 1e-5
                and validation_auc > best_auc
            )
        )

        if improved:
            best_ap = validation_ap
            best_auc = validation_auc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"{label} inner early stopping at epoch {epoch}; "
                f"selected epoch={best_epoch}, "
                f"AUPRC={best_ap:.4f}, "
                f"AUROC={best_auc:.4f}"
            )
            break

    if (
        best_state is None
        or best_epoch < 1
    ):
        raise RuntimeError(
            f"{label}: no valid inner early-stopping checkpoint."
        )

    # Learning rates actually used during epochs 1..best_epoch.
    learning_rate_schedule = [
        float(row["learning_rate"])
        for row in history_rows[
            :best_epoch
        ]
    ]

    # ------------------------------------------------------------------
    # FULL OUTER-TRAINING REFIT
    # ------------------------------------------------------------------
    print(
        f"{label} | STRICT OOF full outer-training refit | "
        f"N={len(y_train):,}, events={int(y_train.sum()):,}, "
        f"epochs={best_epoch}"
    )

    # Reset model to the initialization that existed before inner tuning.
    model.load_state_dict(
        initial_state
    )

    model = model.to(
        device
    )

    # Reset random generators so the refit is reproducible.
    set_seed(seed)

    full_dataset = TabularDataset(
        X_train,
        y_train,
    )

    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate_schedule[0],
        weight_decay=weight_decay,
    )

    for epoch in range(
        1,
        best_epoch + 1,
    ):
        epoch_lr = learning_rate_schedule[
            min(
                epoch - 1,
                len(
                    learning_rate_schedule
                ) - 1,
            )
        ]

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_lr

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in full_loader:
            X_batch = X_batch.to(
                device,
                non_blocking=True,
            )
            y_batch = y_batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(X_batch)
            )

        refit_loss = (
            running_loss
            / len(full_dataset)
        )

        print(
            f"{label} | Refit epoch {epoch:03d}/{best_epoch:03d} | "
            f"Loss={refit_loss:.6f} | "
            f"LR={epoch_lr:.2e}"
        )

    return (
        model,
        pd.DataFrame(
            history_rows
        ),
        {
            "best_epoch": float(
                best_epoch
            ),
            "best_validation_AUPRC": float(
                best_ap
            ),
            "best_validation_AUROC": float(
                best_auc
            ),
            "strict_oof_inner_validation_n": float(
                len(
                    inner_validation_indices
                )
            ),
            "strict_oof_inner_validation_events": float(
                y_inner_validation.sum()
            ),
            "strict_oof_refit_epochs": float(
                best_epoch
            ),
        },
    )


# ============================================================================
# BASELINE FITTING
# ============================================================================


def fit_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=config.logistic_c,
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        max_iter=config.logistic_max_iter,
        n_jobs=config.xgb_n_jobs,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X_train, y_train)
    return model



def fit_asa_only_baseline(
    asa_train: pd.Series,
    y_train: np.ndarray,
    asa_validation: pd.Series,
    asa_test: pd.Series,
    config: PipelineConfig,
    seed: int,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Fit a clinically interpretable ASA-class-only logistic baseline."""
    X_train = pd.DataFrame({"asa": parse_asa_numeric(asa_train)})
    X_validation = pd.DataFrame({"asa": parse_asa_numeric(asa_validation)})
    X_test = pd.DataFrame({"asa": parse_asa_numeric(asa_test)})

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.logistic_c,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=config.logistic_max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    validation_probability = model.predict_proba(X_validation)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]
    return model, validation_probability, test_probability


def make_xgboost_model(
    y_train: np.ndarray,
    config: PipelineConfig,
    seed: int,
):
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Install it or run with --skip_xgboost."
        )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    imbalance_ratio = negatives / max(positives, 1)

    parameters: Dict[str, Any] = {
        "n_estimators": config.xgb_n_estimators,
        "learning_rate": config.xgb_learning_rate,
        "max_depth": config.xgb_max_depth,
        "min_child_weight": config.xgb_min_child_weight,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample_bytree,
        "reg_alpha": config.xgb_reg_alpha,
        "reg_lambda": config.xgb_reg_lambda,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": math.sqrt(imbalance_ratio),
        "tree_method": "hist",
        "max_bin": 256,
        "n_jobs": config.xgb_n_jobs,
        "random_state": seed,
        "verbosity": 0,
    }

    return xgb.XGBClassifier(**parameters)


def fit_xgboost_with_early_stopping(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    early_stopping_rounds: int,
):
    """
    Strict-OOF XGBoost fitting.

    The outer validation fold is never used for early stopping.

    A stratified 20% subset of the outer-training data selects the number of
    trees. A fresh model is then fitted to all outer-training observations
    using that fixed number of estimators.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedShuffleSplit

    y_train = np.asarray(
        y_train,
        dtype=int,
    )

    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=int(
            model.get_params().get(
                "random_state",
                42,
            )
        ),
    )

    inner_train_indices, inner_validation_indices = next(
        inner_splitter.split(
            np.zeros(
                len(y_train)
            ),
            y_train,
        )
    )

    X_inner_train = X_train[
        inner_train_indices
    ]
    y_inner_train = y_train[
        inner_train_indices
    ]

    X_inner_validation = X_train[
        inner_validation_indices
    ]
    y_inner_validation = y_train[
        inner_validation_indices
    ]

    print(
        "XGBoost STRICT OOF inner split | "
        f"train N={len(y_inner_train):,}, "
        f"events={int(y_inner_train.sum()):,} | "
        f"early-stop N={len(y_inner_validation):,}, "
        f"events={int(y_inner_validation.sum()):,}"
    )

    tuning_model = clone(
        model
    )

    # For the inner stopping fit, class weighting is derived only from
    # the inner-training labels. The final refit retains the weighting
    # derived from all outer-training observations.
    inner_negatives = int((y_inner_train == 0).sum())
    inner_positives = int((y_inner_train == 1).sum())

    tuning_model.set_params(
        scale_pos_weight=math.sqrt(
            inner_negatives / max(inner_positives, 1)
        )
    )

    fit_arguments = {
        "X": X_inner_train,
        "y": y_inner_train,
        "eval_set": [
            (
                X_inner_validation,
                y_inner_validation,
            )
        ],
        "verbose": False,
    }

    try:
        tuning_model.fit(
            **fit_arguments,
            early_stopping_rounds=early_stopping_rounds,
        )

    except TypeError:
        tuning_model.set_params(
            early_stopping_rounds=early_stopping_rounds
        )

        tuning_model.fit(
            **fit_arguments
        )

    best_iteration = getattr(
        tuning_model,
        "best_iteration",
        None,
    )

    if best_iteration is None:
        raise RuntimeError(
            "XGBoost strict-OOF inner early stopping did not "
            "produce best_iteration."
        )

    selected_n_estimators = (
        int(best_iteration)
        + 1
    )

    print(
        "XGBoost STRICT OOF full outer-training refit | "
        f"N={len(y_train):,}, "
        f"events={int(y_train.sum()):,}, "
        f"n_estimators={selected_n_estimators}"
    )

    # Fresh full-training fit with the inner-selected tree count.
    model.set_params(
        n_estimators=selected_n_estimators
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    # Preserve audit metadata without affecting XGBoost prediction behavior.
    model.strict_oof_best_iteration_ = int(
        best_iteration
    )

    model.strict_oof_n_estimators_ = int(
        selected_n_estimators
    )

    model.strict_oof_inner_validation_n_ = int(
        len(
            inner_validation_indices
        )
    )

    model.strict_oof_inner_validation_events_ = int(
        y_inner_validation.sum()
    )

    return model


# ============================================================================
# CALIBRATION, THRESHOLDING, AND METRICS
# ============================================================================


def fit_platt_calibrator(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    *,
    c_value: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit a regularized Platt calibrator on model logits.

    A finite C is deliberately used because each held-out fold contains only
    a small number of deaths. This is more stable than an effectively
    unregularized calibrator in rare-event data.
    """
    calibrator = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=2000,
        random_state=int(random_state),
    )
    calibrator.fit(safe_logit(raw_probabilities).reshape(-1, 1), y_true)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        safe_logit(raw_probabilities).reshape(-1, 1)
    )[:, 1]


def cross_fitted_fold_calibration(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    *,
    n_splits: int,
    c_value: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Calibrate one fold without reusing a row to fit its OOF calibrator.

    The base model has never seen the validation fold. A second-level
    stratified cross-fit is used inside that held-out fold to create calibrated
    OOF probabilities. A final calibrator fitted on the complete held-out fold
    is applied to the corresponding fold model's test predictions.
    """
    y_validation = np.asarray(y_validation, dtype=int)
    validation_probabilities = np.asarray(validation_probabilities, dtype=float)
    test_probabilities = np.asarray(test_probabilities, dtype=float)

    class_counts = np.bincount(y_validation, minlength=2)
    usable_splits = int(min(max(2, n_splits), class_counts.min()))

    if usable_splits < 2:
        final_calibrator = fit_platt_calibrator(
            y_validation,
            validation_probabilities,
            c_value=c_value,
            random_state=random_state,
        )
        return (
            apply_platt_calibrator(final_calibrator, validation_probabilities),
            apply_platt_calibrator(final_calibrator, test_probabilities),
            final_calibrator,
        )

    calibrated_validation = np.full(len(y_validation), np.nan, dtype=float)
    calibration_cv = StratifiedKFold(
        n_splits=usable_splits,
        shuffle=True,
        random_state=random_state,
    )

    logits = safe_logit(validation_probabilities).reshape(-1, 1)
    for calibration_train, calibration_holdout in calibration_cv.split(
        logits, y_validation
    ):
        calibrator = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=2000,
            random_state=int(random_state),
        )
        calibrator.fit(logits[calibration_train], y_validation[calibration_train])
        calibrated_validation[calibration_holdout] = calibrator.predict_proba(
            logits[calibration_holdout]
        )[:, 1]

    if not np.isfinite(calibrated_validation).all():
        raise RuntimeError("Cross-fitted Platt calibration produced missing values.")

    final_calibrator = fit_platt_calibrator(
        y_validation,
        validation_probabilities,
        c_value=c_value,
        random_state=random_state,
    )
    calibrated_test = apply_platt_calibrator(
        final_calibrator, test_probabilities
    )
    return calibrated_validation, calibrated_test, final_calibrator


def optimize_darn_xgb_blend(
    y_true: np.ndarray,
    darn_probabilities: np.ndarray,
    xgb_probabilities: np.ndarray,
    *,
    grid_size: int = 181,
    min_darn_weight: float = 0.10,
    max_darn_weight: float = 0.90,
) -> Tuple[float, pd.DataFrame]:
    """Select a DARN weight using development OOF predictions only."""
    if grid_size < 2:
        raise ValueError("hybrid_blend_grid_size must be at least 2.")
    if not (0.0 <= min_darn_weight <= max_darn_weight <= 1.0):
        raise ValueError("Hybrid DARN weight bounds must satisfy 0 <= min <= max <= 1.")

    darn_probabilities = np.asarray(darn_probabilities, dtype=float)
    xgb_probabilities = np.asarray(xgb_probabilities, dtype=float)
    weights = np.linspace(min_darn_weight, max_darn_weight, grid_size)

    rows: List[Dict[str, float]] = []
    best_key: Optional[Tuple[float, float, float]] = None
    best_weight = float(weights[0])

    for darn_weight in weights:
        blended = (
            float(darn_weight) * darn_probabilities
            + (1.0 - float(darn_weight)) * xgb_probabilities
        )
        auprc = float(average_precision_score(y_true, blended))
        auroc = float(roc_auc_score(y_true, blended))
        brier = float(brier_score_loss(y_true, blended))
        rows.append(
            {
                "darn_weight": float(darn_weight),
                "xgboost_weight": float(1.0 - darn_weight),
                "OOF_AUPRC": auprc,
                "OOF_AUROC": auroc,
                "OOF_Brier": brier,
            }
        )

        # Primary objective: AUPRC. Tie-break with AUROC, then lower Brier.
        key = (auprc, auroc, -brier)
        if best_key is None or key > best_key:
            best_key = key
            best_weight = float(darn_weight)

    return best_weight, pd.DataFrame(rows)


def make_darn_xgb_stack_features(
    darn_probabilities: np.ndarray,
    xgb_probabilities: np.ndarray,
) -> np.ndarray:
    """Compact meta-features from two independently cross-fitted base models."""
    darn_logit = safe_logit(darn_probabilities)
    xgb_logit = safe_logit(xgb_probabilities)
    return np.column_stack(
        [
            darn_logit,
            xgb_logit,
            darn_logit - xgb_logit,
            0.5 * (darn_logit + xgb_logit),
            np.abs(darn_logit - xgb_logit),
        ]
    ).astype(np.float64)


def cross_fitted_darn_xgb_stacker(
    y_true: np.ndarray,
    darn_oof: np.ndarray,
    xgb_oof: np.ndarray,
    darn_test: np.ndarray,
    xgb_test: np.ndarray,
    *,
    n_folds: int,
    c_value: float,
    class_weight_balanced: bool,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Generate second-level OOF predictions and a final test prediction."""
    development_features = make_darn_xgb_stack_features(darn_oof, xgb_oof)
    test_features = make_darn_xgb_stack_features(darn_test, xgb_test)

    stack_oof = np.full(len(y_true), np.nan, dtype=float)
    meta_cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    class_weight = "balanced" if class_weight_balanced else None

    for meta_train, meta_validation in meta_cv.split(
        development_features, y_true
    ):
        stacker = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )
        stacker.fit(
            development_features[meta_train],
            y_true[meta_train],
        )
        stack_oof[meta_validation] = stacker.predict_proba(
            development_features[meta_validation]
        )[:, 1]

    if not np.isfinite(stack_oof).all():
        raise RuntimeError("Hybrid stacker produced missing OOF predictions.")

    final_stacker = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight=class_weight,
        random_state=random_state,
    )
    final_stacker.fit(development_features, y_true)
    stack_test = final_stacker.predict_proba(test_features)[:, 1]

    return stack_oof, stack_test, final_stacker


def select_operating_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    strategy: str = "sensitivity",
    target_sensitivity: float = 0.80,
    target_specificity: float = 0.95,
) -> float:
    strategy = strategy.lower().strip()

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        finite = np.isfinite(thresholds)
        if not finite.any():
            return 0.5
        index = np.argmax((tpr - fpr)[finite])
        return float(thresholds[finite][index])

    if strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        if len(thresholds) == 0:
            return 0.5
        f1_values = 2.0 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        return float(thresholds[int(np.nanargmax(f1_values))])

    if strategy == "specificity":
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        finite = np.isfinite(thresholds)
        specificity = 1.0 - fpr
        eligible = np.where((specificity >= target_specificity) & finite)[0]
        if len(eligible) == 0:
            return float(np.nanmedian(probabilities))
        # Among thresholds meeting specificity, maximize sensitivity; break ties with lower threshold.
        best_tpr = np.max(tpr[eligible])
        candidates = eligible[tpr[eligible] == best_tpr]
        selected = candidates[np.argmin(thresholds[candidates])]
        return float(thresholds[selected])

    if strategy == "sensitivity":
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        finite = np.isfinite(thresholds)
        eligible = np.where((tpr >= target_sensitivity) & finite)[0]
        if len(eligible) == 0:
            return float(np.nanmedian(probabilities))

        # Among thresholds meeting sensitivity, choose the one with minimum FPR.
        minimum_fpr = np.min(fpr[eligible])
        candidates = eligible[fpr[eligible] == minimum_fpr]
        selected = candidates[np.argmax(thresholds[candidates])]
        return float(thresholds[selected])

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def threshold_from_alert_fraction(
    probabilities: np.ndarray,
    alert_fraction: float,
) -> float:
    """Return a threshold that flags approximately the requested top fraction."""
    probabilities = np.asarray(probabilities, dtype=float)
    alert_fraction = float(np.clip(alert_fraction, 1.0 / max(len(probabilities), 1), 1.0))
    sorted_probabilities = np.sort(probabilities)[::-1]
    count = int(np.clip(math.ceil(alert_fraction * len(probabilities)), 1, len(probabilities)))
    return float(sorted_probabilities[count - 1])


def transport_threshold_to_test(
    oof_probabilities: np.ndarray,
    development_threshold: float,
    test_probabilities: np.ndarray,
    *,
    mode: str = "auto",
    max_rate_ratio: float = 3.0,
    max_absolute_gap: float = 0.25,
    degenerate_rate: float = 0.99,
) -> Tuple[float, Dict[str, Any]]:
    """Transport a development operating point without using test labels.

    The absolute calibrated-probability threshold is used whenever its alert
    rate remains compatible with development. If averaging several fold models
    causes a severe score-distribution shift, ``auto`` transports the
    development alert fraction to the test score distribution. This prevents
    degenerate all-positive/all-negative classifications while preserving the
    prespecified development alert burden.
    """
    mode = str(mode).lower().strip()
    if mode not in {"absolute", "alert_rate", "auto"}:
        raise ValueError(
            "threshold_transport_mode must be 'absolute', 'alert_rate', or 'auto'."
        )

    oof_probabilities = np.asarray(oof_probabilities, dtype=float)
    test_probabilities = np.asarray(test_probabilities, dtype=float)
    oof_alert_rate = float(np.mean(oof_probabilities >= development_threshold))
    absolute_test_alert_rate = float(
        np.mean(test_probabilities >= development_threshold)
    )

    lower_expected = oof_alert_rate / max(max_rate_ratio, 1.0)
    upper_expected = min(1.0, oof_alert_rate * max(max_rate_ratio, 1.0))
    severe_shift = (
        absolute_test_alert_rate >= degenerate_rate
        or absolute_test_alert_rate <= (1.0 - degenerate_rate)
        or absolute_test_alert_rate < lower_expected
        or absolute_test_alert_rate > upper_expected
        or abs(absolute_test_alert_rate - oof_alert_rate) > max_absolute_gap
    )

    use_alert_rate = mode == "alert_rate" or (mode == "auto" and severe_shift)
    if use_alert_rate:
        applied_threshold = threshold_from_alert_fraction(
            test_probabilities, oof_alert_rate
        )
        method = "development_alert_rate_quantile"
    else:
        applied_threshold = float(development_threshold)
        method = "absolute_calibrated_probability"

    applied_test_alert_rate = float(
        np.mean(test_probabilities >= applied_threshold)
    )
    metadata = {
        "development_threshold": float(development_threshold),
        "applied_test_threshold": float(applied_threshold),
        "development_alert_rate": oof_alert_rate,
        "test_alert_rate_at_absolute_threshold": absolute_test_alert_rate,
        "applied_test_alert_rate": applied_test_alert_rate,
        "transport_method": method,
        "severe_distribution_shift_detected": bool(severe_shift),
    }
    return float(applied_threshold), metadata


def calibration_slope_intercept(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, float]:
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(safe_logit(probabilities).reshape(-1, 1), y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(probabilities, quantiles))
    if len(edges) <= 2:
        return float(abs(probabilities.mean() - y_true.mean()))

    bin_ids = np.digitize(probabilities, edges[1:-1], right=True)
    ece = 0.0

    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        ece += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())

    return float(ece)



def hosmer_lemeshow_test(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_groups: int = 10,
) -> Tuple[float, int, float]:
    """Hosmer-Lemeshow statistic using quantile groups; interpreted descriptively."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1.0 - 1e-10)
    try:
        groups = pd.qcut(p, q=n_groups, duplicates="drop")
    except ValueError:
        return np.nan, 0, np.nan

    table = pd.DataFrame({"y": y, "p": p, "group": groups}).groupby(
        "group", observed=True
    ).agg(observed=("y", "sum"), expected=("p", "sum"), n=("y", "size"))

    if len(table) < 3:
        return np.nan, 0, np.nan

    expected = np.clip(table["expected"].to_numpy(float), 1e-8, None)
    expected_non = np.clip(table["n"].to_numpy(float) - expected, 1e-8, None)
    observed = table["observed"].to_numpy(float)
    observed_non = table["n"].to_numpy(float) - observed
    statistic = float(np.sum((observed - expected) ** 2 / expected + (observed_non - expected_non) ** 2 / expected_non))
    degrees_freedom = max(int(len(table) - 2), 1)
    p_value = float(chi2.sf(statistic, degrees_freedom))
    return statistic, degrees_freedom, p_value


def build_risk_concentration_table(
    y_true: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_events = int(np.asarray(y_true).sum())
    for model_name, probabilities in model_probabilities.items():
        order = np.argsort(probabilities)[::-1]
        for fraction in fractions:
            count = max(1, int(math.ceil(len(y_true) * fraction)))
            selected = order[:count]
            captured = int(np.asarray(y_true)[selected].sum())
            rows.append(
                {
                    "model": model_name,
                    "risk_fraction": float(fraction),
                    "risk_percent": int(round(100 * fraction)),
                    "patients_flagged": int(count),
                    "events_captured": captured,
                    "event_capture_fraction": float(captured / max(total_events, 1)),
                    "PPV_within_stratum": float(captured / count),
                    "number_needed_to_evaluate": float(count / captured) if captured > 0 else np.inf,
                }
            )
    return pd.DataFrame(rows)


def decision_curve_analysis(
    y_true: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    min_threshold: float,
    max_threshold: float,
    n_points: int,
) -> pd.DataFrame:
    """Compute standardized clinical net benefit without using thresholds for training."""
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    prevalence = float(y.mean())
    thresholds = np.geomspace(min_threshold, max_threshold, n_points)
    rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        odds = threshold / max(1.0 - threshold, 1e-12)
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows.append({"model": "Treat none", "threshold_probability": threshold, "net_benefit": 0.0})
        rows.append({"model": "Treat all", "threshold_probability": threshold, "net_benefit": treat_all})

        for model_name, probabilities in model_probabilities.items():
            predicted = probabilities >= threshold
            tp = int(np.sum(predicted & (y == 1)))
            fp = int(np.sum(predicted & (y == 0)))
            net_benefit = tp / n - fp / n * odds
            rows.append(
                {
                    "model": model_name,
                    "threshold_probability": float(threshold),
                    "net_benefit": float(net_benefit),
                    "TP": tp,
                    "FP": fp,
                    "patients_flagged": int(predicted.sum()),
                }
            )
    return pd.DataFrame(rows)


def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    prevalence = float(y_true.mean())
    slope, intercept = calibration_slope_intercept(y_true, probabilities)
    hl_statistic, hl_df, hl_p = hosmer_lemeshow_test(y_true, probabilities, n_groups=10)

    metrics: Dict[str, float] = {
        "N": float(len(y_true)),
        "Events": float(y_true.sum()),
        "Prevalence": prevalence,
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "AUPRC_fold_over_baseline": float(
            average_precision_score(y_true, probabilities) / max(prevalence, 1e-12)
        ),
        "Brier": float(brier_score_loss(y_true, probabilities)),
        "Calibration_slope": slope,
        "Calibration_intercept": intercept,
        "ECE": expected_calibration_error(y_true, probabilities),
        "Hosmer_Lemeshow_chi2": float(hl_statistic),
        "Hosmer_Lemeshow_df": float(hl_df),
        "Hosmer_Lemeshow_p": float(hl_p),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "Specificity": float(specificity),
        "False_positive_rate": float(1.0 - specificity),
        "False_negative_rate": float(fn / max(fn + tp, 1)),
        "NPV": float(npv),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Threshold": float(threshold),
        "TP": float(tp),
        "FN": float(fn),
        "FP": float(fp),
        "TN": float(tn),
    }

    for fraction in (0.01, 0.02, 0.05, 0.10):
        count = max(1, int(math.ceil(len(y_true) * fraction)))
        top_indices = np.argsort(probabilities)[::-1][:count]
        captured = int(y_true[top_indices].sum())
        metrics[f"Top_{int(fraction * 100)}pct_N"] = float(count)
        metrics[f"Top_{int(fraction * 100)}pct_events"] = float(captured)
        metrics[f"Top_{int(fraction * 100)}pct_event_capture"] = float(
            captured / max(y_true.sum(), 1)
        )
        metrics[f"Top_{int(fraction * 100)}pct_PPV"] = float(captured / count)
        metrics[f"Top_{int(fraction * 100)}pct_NNE"] = float(count / captured) if captured > 0 else float("inf")

    return metrics


# ============================================================================
# BOOTSTRAP INFERENCE
# ============================================================================


def stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]

    sampled_positive = rng.choice(
        positive_indices, size=len(positive_indices), replace=True
    )
    sampled_negative = rng.choice(
        negative_indices, size=len(negative_indices), replace=True
    )

    sampled = np.concatenate([sampled_positive, sampled_negative])
    rng.shuffle(sampled)
    return sampled


def bootstrap_model_metrics(
    y_true: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    n_bootstrap: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    rng = np.random.default_rng(random_state)
    metric_names = [
        "AUROC", "AUPRC", "Brier", "Accuracy", "Balanced_Accuracy",
        "Recall", "Specificity", "Precision",
    ]
    distributions: Dict[str, Dict[str, List[float]]] = {
        model: {metric: [] for metric in metric_names}
        for model in model_probabilities
    }

    for _ in range(n_bootstrap):
        indices = stratified_bootstrap_indices(y_true, rng)
        y_bootstrap = y_true[indices]

        for model_name, probabilities in model_probabilities.items():
            p = probabilities[indices]
            pred = (p >= thresholds[model_name]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_bootstrap, pred, labels=[0, 1]).ravel()
            distributions[model_name]["AUROC"].append(roc_auc_score(y_bootstrap, p))
            distributions[model_name]["AUPRC"].append(average_precision_score(y_bootstrap, p))
            distributions[model_name]["Brier"].append(brier_score_loss(y_bootstrap, p))
            distributions[model_name]["Accuracy"].append(accuracy_score(y_bootstrap, pred))
            distributions[model_name]["Balanced_Accuracy"].append(balanced_accuracy_score(y_bootstrap, pred))
            distributions[model_name]["Recall"].append(tp / max(tp + fn, 1))
            distributions[model_name]["Specificity"].append(tn / max(tn + fp, 1))
            distributions[model_name]["Precision"].append(tp / max(tp + fp, 1))

    rows: List[Dict[str, Any]] = []
    array_distributions: Dict[str, Dict[str, np.ndarray]] = {}
    point_metrics = {
        model: calculate_binary_metrics(y_true, p, thresholds[model])
        for model, p in model_probabilities.items()
    }

    for model_name, metric_values in distributions.items():
        array_distributions[model_name] = {}
        for metric_name, values in metric_values.items():
            array = np.asarray(values, dtype=float)
            array_distributions[model_name][metric_name] = array
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "estimate": float(point_metrics[model_name][metric_name]),
                    "CI_low": float(np.quantile(array, 0.025)),
                    "CI_high": float(np.quantile(array, 0.975)),
                }
            )

    return pd.DataFrame(rows), array_distributions

def paired_bootstrap_comparisons(
    bootstrap_distributions: Dict[str, Dict[str, np.ndarray]],
    reference_model: str = "Uniform DARN",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if reference_model not in bootstrap_distributions:
        return pd.DataFrame()

    for comparator in bootstrap_distributions:
        if comparator == reference_model:
            continue

        for metric in ("AUROC", "AUPRC", "Brier"):
            reference = bootstrap_distributions[reference_model][metric]
            comparison = bootstrap_distributions[comparator][metric]

            # Positive difference favors DARN for AUROC/AUPRC; for Brier,
            # reverse the sign so positive still favors DARN.
            if metric == "Brier":
                differences = comparison - reference
            else:
                differences = reference - comparison

            p_two_sided = min(
                1.0,
                2.0 * min(
                    float(np.mean(differences <= 0.0)),
                    float(np.mean(differences >= 0.0)),
                ),
            )

            rows.append(
                {
                    "reference_model": reference_model,
                    "comparator": comparator,
                    "metric": metric,
                    "difference_favoring_reference": float(differences.mean()),
                    "CI_low": float(np.quantile(differences, 0.025)),
                    "CI_high": float(np.quantile(differences, 0.975)),
                    "bootstrap_p_two_sided": float(p_two_sided),
                }
            )

    return pd.DataFrame(rows)



# ============================================================================
# SUBGROUP FAIRNESS, ERROR ANALYSIS, AND CORRELATION AUDITS
# ============================================================================


def derive_subgroup_variables(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Prespecified subgroup definitions used in the revised manuscript."""
    groups = pd.DataFrame(index=X_raw.index)

    if "sex" in X_raw.columns:
        raw = X_raw["sex"].astype("string").str.strip().str.lower()
        numeric = pd.to_numeric(X_raw["sex"], errors="coerce")
        sex = pd.Series("Missing", index=X_raw.index, dtype="object")
        sex.loc[
            raw.isin(["f", "female", "woman", "women", "false", "0"])
            | raw.str.contains("female", na=False)
            | numeric.eq(0)
        ] = "Female"
        sex.loc[
            raw.isin(["m", "male", "man", "men", "true", "1"])
            | (
                raw.str.contains("male", na=False)
                & ~raw.str.contains("female", na=False)
            )
            | numeric.eq(1)
        ] = "Male"
        groups["Sex"] = sex.astype("string")

    if "age" in X_raw.columns:
        age = pd.to_numeric(X_raw["age"], errors="coerce")
        groups["Age"] = pd.cut(
            age,
            bins=[-np.inf, 60.0, 70.0, np.inf],
            labels=["<60", "60-69", ">=70"],
            right=False,
        ).astype("string").fillna("Missing")

    if "race" in X_raw.columns:
        groups["Race"] = X_raw["race"].astype("string").fillna("Missing")

    if "asa" in X_raw.columns:
        asa = parse_asa_numeric(X_raw["asa"])
        asa_group = pd.Series("Missing", index=X_raw.index, dtype="object")
        asa_group.loc[asa.isin([1, 2])] = "ASA I-II"
        asa_group.loc[asa.eq(3)] = "ASA III"
        asa_group.loc[asa.isin([4, 5])] = "ASA IV-V"
        groups["ASA"] = asa_group.astype("string")

    if "emop" in X_raw.columns:
        emergency = pd.to_numeric(X_raw["emop"], errors="coerce")
        groups["Urgency"] = (
            emergency.map({0.0: "Elective", 1.0: "Emergency"})
            .fillna("Missing")
            .astype("string")
        )

    return groups


def _subgroup_point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    result = {
        "N": float(len(y)),
        "Events": float(y.sum()),
        "Nonevents": float((y == 0).sum()),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(precision),
        "NPV": float(npv),
        "FPR": float(1.0 - specificity),
        "FNR": float(1.0 - sensitivity),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
        "Brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) == 2:
        result["AUROC"] = float(roc_auc_score(y, p))
        result["AUPRC"] = float(average_precision_score(y, p))
    else:
        result["AUROC"] = np.nan
        result["AUPRC"] = np.nan
    return result


def subgroup_performance_with_bootstrap(
    y_true: np.ndarray,
    probabilities: Dict[str, np.ndarray],
    thresholds_by_operating_point: Dict[str, Dict[str, float]],
    subgroup_frame: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
    min_events: int,
    min_nonevents: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    metrics_for_ci = ["Sensitivity", "Specificity", "PPV", "NPV", "FPR", "FNR", "Accuracy", "Balanced_Accuracy", "AUROC", "AUPRC", "Brier"]

    for attribute in subgroup_frame.columns:
        values = subgroup_frame[attribute].astype("string").fillna("Missing")
        for level in sorted(values.unique().tolist()):
            mask = values.eq(level).to_numpy()
            y_group = np.asarray(y_true)[mask]
            if len(y_group) == 0:
                continue

            positive_indices = np.where(y_group == 1)[0]
            negative_indices = np.where(y_group == 0)[0]

            for operating_point, model_thresholds in thresholds_by_operating_point.items():
                for model_name, p_all in probabilities.items():
                    p_group = np.asarray(p_all)[mask]
                    threshold = model_thresholds[model_name]
                    point = _subgroup_point_metrics(y_group, p_group, threshold)
                    row: Dict[str, Any] = {
                        "model": model_name,
                        "operating_point": operating_point,
                        "attribute": attribute,
                        "group": str(level),
                        **point,
                    }

                    if len(positive_indices) >= min_events and len(negative_indices) >= min_nonevents:
                        bootstrap_values = {metric: [] for metric in metrics_for_ci}
                        for _ in range(n_bootstrap):
                            sampled_pos = rng.choice(positive_indices, size=len(positive_indices), replace=True)
                            sampled_neg = rng.choice(negative_indices, size=len(negative_indices), replace=True)
                            sampled = np.concatenate([sampled_pos, sampled_neg])
                            rng.shuffle(sampled)
                            boot = _subgroup_point_metrics(y_group[sampled], p_group[sampled], threshold)
                            for metric in metrics_for_ci:
                                bootstrap_values[metric].append(boot.get(metric, np.nan))

                        for metric, values_boot in bootstrap_values.items():
                            arr = np.asarray(values_boot, dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if len(arr):
                                row[f"{metric}_CI_low"] = float(np.quantile(arr, 0.025))
                                row[f"{metric}_CI_high"] = float(np.quantile(arr, 0.975))
                    else:
                        row["CI_note"] = f"Insufficient events/non-events for stable bootstrap CI (<{min_events} events or <{min_nonevents} non-events)"

                    rows.append(row)

    performance = pd.DataFrame(rows)
    gap_rows: List[Dict[str, Any]] = []
    eligible = performance[
        (performance["Events"] >= min_events)
        & (performance["Nonevents"] >= min_nonevents)
    ].copy()

    for (model_name, operating_point, attribute), group in eligible.groupby(
        ["model", "operating_point", "attribute"], dropna=False
    ):
        if len(group) < 2:
            continue
        sensitivity_gap = float(group["Sensitivity"].max() - group["Sensitivity"].min())
        specificity_gap = float(group["Specificity"].max() - group["Specificity"].min())
        fpr_gap = float(group["FPR"].max() - group["FPR"].min())
        ppv_gap = float(group["PPV"].max() - group["PPV"].min())
        gap_rows.append(
            {
                "model": model_name,
                "operating_point": operating_point,
                "attribute": attribute,
                "n_eligible_groups": int(len(group)),
                "sensitivity_gap": sensitivity_gap,
                "specificity_gap": specificity_gap,
                "FPR_gap": fpr_gap,
                "PPV_gap": ppv_gap,
                "equalized_odds_difference": float(max(sensitivity_gap, fpr_gap)),
            }
        )

    return performance, pd.DataFrame(gap_rows)


def create_error_analysis(
    X_test_raw: pd.DataFrame,
    y_true: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    primary_model: str,
    primary_threshold: float,
    selective_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability = np.asarray(model_probabilities[primary_model], dtype=float)
    prediction = (probability >= primary_threshold).astype(int)
    selective_prediction = (probability >= selective_threshold).astype(int)
    error_type = np.select(
        [
            (y_true == 1) & (prediction == 1),
            (y_true == 1) & (prediction == 0),
            (y_true == 0) & (prediction == 1),
            (y_true == 0) & (prediction == 0),
        ],
        ["True positive", "False negative", "False positive", "True negative"],
        default="Unknown",
    )

    preferred_columns = [
        "age", "sex", "race", "asa", "bmi", "emop",
        "surgery_duration", "or_duration", "anesthesia_duration",
    ]
    additional_patterns = [r"intraop_.*ebl", r"preop_.*albumin", r"preop_.*platelet", r"preop_.*creatinine"]
    selected_columns = [column for column in preferred_columns if column in X_test_raw.columns]
    for pattern in additional_patterns:
        matches = [column for column in X_test_raw.columns if re.search(pattern, column, flags=re.IGNORECASE)]
        selected_columns.extend(matches[:2])
    selected_columns = list(dict.fromkeys(selected_columns))

    cases = X_test_raw.loc[:, selected_columns].copy()
    cases.insert(0, "error_type", error_type)
    cases.insert(0, "selective_prediction", selective_prediction)
    cases.insert(0, "high_sensitivity_prediction", prediction)
    cases.insert(0, "predicted_probability", probability)
    cases.insert(0, "y_true", y_true)

    if "Uniform DARN" in model_probabilities and "XGBoost" in model_probabilities:
        darn = np.asarray(model_probabilities["Uniform DARN"])
        xgb = np.asarray(model_probabilities["XGBoost"])
        cases["darn_probability"] = darn
        cases["xgboost_probability"] = xgb
        cases["darn_xgb_absolute_disagreement"] = np.abs(darn - xgb)

    numeric_columns = cases.select_dtypes(include=np.number).columns.difference(
        ["y_true", "high_sensitivity_prediction", "selective_prediction"]
    )
    numeric_rows: List[Dict[str, Any]] = []
    for error_level, group in cases.groupby("error_type"):
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values) == 0:
                continue
            numeric_rows.append(
                {
                    "error_type": error_level,
                    "variable": column,
                    "N": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "median": float(values.median()),
                    "Q1": float(values.quantile(0.25)),
                    "Q3": float(values.quantile(0.75)),
                }
            )

    categorical_rows: List[Dict[str, Any]] = []
    categorical_columns = [column for column in selected_columns if column not in numeric_columns]
    for error_level, group in cases.groupby("error_type"):
        for column in categorical_columns:
            counts = group[column].astype("string").fillna("Missing").value_counts(dropna=False)
            for level, count in counts.items():
                categorical_rows.append(
                    {
                        "error_type": error_level,
                        "variable": column,
                        "level": str(level),
                        "count": int(count),
                        "percent_within_error_type": float(100.0 * count / max(len(group), 1)),
                    }
                )

    return cases, pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)


def save_training_correlation_audit(
    X_train: np.ndarray,
    selected_feature_names: np.ndarray,
    selected_domain_names: Sequence[str],
    selected_domain_indices: Dict[str, np.ndarray],
    feature_scores: pd.DataFrame,
    top_n: int,
    output_directory: pathlib.Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    score_map = dict(zip(feature_scores["feature"].astype(str), feature_scores["selection_score"]))
    scores = np.asarray([score_map.get(str(name), 0.0) for name in selected_feature_names])
    top_indices = np.argsort(scores)[::-1][: min(top_n, X_train.shape[1])]
    top_names = [str(selected_feature_names[index]) for index in top_indices]
    top_frame = pd.DataFrame(X_train[:, top_indices], columns=top_names)
    correlation = top_frame.corr(method="spearman").fillna(0.0)
    correlation.to_csv(output_directory / "feature_spearman_correlation_top_selected.csv")

    if len(top_names) >= 3:
        distance = 1.0 - np.abs(correlation.to_numpy())
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        clustering = linkage(condensed, method="average")
        plt.figure(figsize=(10, max(7, 0.28 * len(top_names))))
        dendrogram(clustering, labels=top_names, orientation="left", leaf_font_size=7)
        plt.title("Training-fold feature clustering (absolute Spearman distance)")
        plt.xlabel("Distance")
        plt.tight_layout()
        plt.savefig(output_directory / "feature_dendrogram_horizontal.png", dpi=400, bbox_inches="tight")
        plt.close()

    domain_aggregates: Dict[str, np.ndarray] = {}
    for domain in selected_domain_names:
        indices = selected_domain_indices[domain]
        domain_aggregates[domain] = np.nanmean(X_train[:, indices], axis=1)
    pd.DataFrame(domain_aggregates).corr(method="spearman").to_csv(
        output_directory / "domain_aggregate_spearman_correlation.csv"
    )


# ============================================================================
# PLOTS
# ============================================================================


def save_comparison_plots(
    y_true: np.ndarray,
    probabilities: Dict[str, np.ndarray],
    metrics_table: pd.DataFrame,
    figure_directory: pathlib.Path,
    calibration_bootstrap: int = 500,
    calibration_bins: int = 10,
) -> None:
    figure_directory.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(7, 7))
    for model_name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mortality Model ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_directory / "ROC_all_models.png", dpi=400)
    plt.close()

    # Precision-recall
    prevalence = float(y_true.mean())
    plt.figure(figsize=(7, 7))
    for model_name, p in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} ({ap:.3f})")
    plt.axhline(prevalence, linestyle="--", linewidth=1, label=f"Baseline ({prevalence:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mortality Model Precision-Recall Curves")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_directory / "PR_all_models.png", dpi=400)
    plt.close()

    # Calibration
    plt.figure(figsize=(7, 7))
    for model_name, p in probabilities.items():
        observed, predicted = calibration_curve(
            y_true,
            p,
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(predicted, observed, marker="o", linewidth=2, label=model_name)
    upper = max(0.01, max(float(np.quantile(p, 0.99)) for p in probabilities.values()))
    plt.plot([0, upper], [0, upper], linestyle="--", linewidth=1)
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Mortality Rate")
    plt.title("Calibration Curves")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_directory / "Calibration_all_models.png", dpi=400)
    plt.close()


    # Individual reliability diagrams with stratified-bootstrap confidence bands.
    rng = np.random.default_rng(20260716)
    for model_name, p in probabilities.items():
        p = np.asarray(p, dtype=float)
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, calibration_bins + 1)))
        if len(edges) < 3:
            continue
        bin_ids = np.digitize(p, edges[1:-1], right=True)
        x_values: List[float] = []
        observed_values: List[float] = []
        lower_values: List[float] = []
        upper_values: List[float] = []
        bootstrap_observed: List[List[float]] = [[] for _ in range(len(edges) - 1)]

        for _ in range(calibration_bootstrap):
            indices = stratified_bootstrap_indices(np.asarray(y_true), rng)
            y_boot = np.asarray(y_true)[indices]
            p_boot = p[indices]
            boot_bins = np.digitize(p_boot, edges[1:-1], right=True)
            for bin_id in range(len(edges) - 1):
                mask = boot_bins == bin_id
                if mask.any():
                    bootstrap_observed[bin_id].append(float(y_boot[mask].mean()))

        for bin_id in range(len(edges) - 1):
            mask = bin_ids == bin_id
            if not mask.any():
                continue
            x_values.append(float(p[mask].mean()))
            observed_values.append(float(np.asarray(y_true)[mask].mean()))
            values = np.asarray(bootstrap_observed[bin_id], dtype=float)
            lower_values.append(float(np.quantile(values, 0.025)) if len(values) else np.nan)
            upper_values.append(float(np.quantile(values, 0.975)) if len(values) else np.nan)

        upper_axis = max(0.01, float(np.nanmax(upper_values + x_values)) * 1.08)
        plt.figure(figsize=(7, 7))
        plt.plot(x_values, observed_values, marker="o", linewidth=2, label=model_name)
        plt.fill_between(x_values, lower_values, upper_values, alpha=0.20, label="95% bootstrap CI")
        plt.plot([0, upper_axis], [0, upper_axis], linestyle="--", linewidth=1)
        plt.xlim(0, upper_axis)
        plt.ylim(0, upper_axis)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed mortality rate")
        plt.title(f"Reliability diagram: {model_name}")
        plt.legend(loc="upper left")
        plt.tight_layout()
        safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
        plt.savefig(figure_directory / f"Calibration_{safe_model}_with_CI.png", dpi=400)
        plt.close()

    # Metric bars
    ordered = metrics_table.sort_values("AUPRC", ascending=False)
    x_positions = np.arange(len(ordered))

    plt.figure(figsize=(9, 6))
    plt.bar(x_positions, ordered["AUROC"].values)
    plt.xticks(x_positions, ordered["model"].values, rotation=20, ha="right")
    plt.ylabel("AUROC")
    plt.ylim(max(0.0, ordered["AUROC"].min() - 0.05), 1.0)
    plt.title("Test AUROC by Model")
    plt.tight_layout()
    plt.savefig(figure_directory / "AUROC_bar.png", dpi=400)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.bar(x_positions, ordered["AUPRC"].values)
    plt.xticks(x_positions, ordered["model"].values, rotation=20, ha="right")
    plt.ylabel("AUPRC")
    plt.title("Test AUPRC by Model")
    plt.tight_layout()
    plt.savefig(figure_directory / "AUPRC_bar.png", dpi=400)
    plt.close()


# ============================================================================
# OPTIONAL DIRECT DARN GRADIENTSHAP PER FOLD
# ============================================================================


def aggregate_gradient_shap_for_fold(
    model: UniformDARNClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: np.ndarray,
    max_samples: int,
    n_samples: int,
    seed: int,
) -> Optional[pd.DataFrame]:
    try:
        from captum.attr import GradientShap
    except Exception:
        print("Captum is unavailable; skipping GradientSHAP.")
        return None

    rng = np.random.default_rng(seed)
    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]

    remaining = max(0, max_samples - len(positive_indices))
    selected_negative = rng.choice(
        negative_indices,
        size=min(remaining, len(negative_indices)),
        replace=False,
    )
    selected_indices = np.concatenate([positive_indices, selected_negative])
    rng.shuffle(selected_indices)

    attribution_X = torch.as_tensor(
        X_test[selected_indices], dtype=torch.float32
    )

    baseline_count = min(128, len(X_train))
    baseline_indices = rng.choice(
        np.arange(len(X_train)),
        size=baseline_count,
        replace=False,
    )
    baselines = torch.as_tensor(
        X_train[baseline_indices], dtype=torch.float32
    )

    device = next(model.parameters()).device
    attribution_X = attribution_X.to(device)
    baselines = baselines.to(device)
    model.eval()

    explainer = GradientShap(model)
    attributions = explainer.attribute(
        attribution_X,
        baselines=baselines,
        n_samples=n_samples,
        stdevs=0.0,
    )

    importance = np.abs(attributions.detach().cpu().numpy()).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_gradientshap": importance,
        }
    )



# ============================================================================
# TRUE LEAVE-ONE-DOMAIN-OUT RETRAINING AND TARGETED REVIEWER ANALYSES
# ============================================================================


def _safe_variant_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def raw_columns_removed_for_domain(
    raw_feature_columns: Sequence[str],
    domain: str,
) -> List[str]:
    """Return direct and dependency-aware raw columns for domain removal.

    Interactions derived from a removed source domain are also removed so that
    information from age, ASA, emergency status, or duration cannot leak back
    into a leave-one-domain-out model through prespecified interaction terms.
    """
    domain = str(domain)
    columns = [str(column) for column in raw_feature_columns]
    removed = {
        column
        for column in columns
        if assign_raw_feature_domain(column) == domain
    }

    if domain == "Demographics":
        for column in columns:
            lowered = column.lower()
            if column.startswith("INT_") and any(
                token in lowered
                for token in ("age", "asa", "emergency")
            ):
                removed.add(column)

    if domain == "Durations":
        for column in columns:
            lowered = column.lower()
            if column.startswith("INT_") and "duration" in lowered:
                removed.add(column)

    return sorted(removed)


def _calibration_absolute_oe_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    observed = float(np.asarray(y_true, dtype=int).sum())
    expected = float(np.asarray(probabilities, dtype=float).sum())
    if expected <= 0.0:
        return np.nan
    return float(abs(observed / expected - 1.0))


def paired_bootstrap_retraining_difference(
    y_true: np.ndarray,
    full_probabilities: np.ndarray,
    retrained_probabilities: np.ndarray,
    *,
    analysis: str,
    removed_feature_set: str,
    model_name: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    """Paired stratified bootstrap differences on the same holdout rows.

    Positive differences always favor the complete model:
      - AUROC/AUPRC: complete minus retrained.
      - Brier: retrained minus complete.
      - absolute O/E error: retrained minus complete.
    """
    y_true = np.asarray(y_true, dtype=int)
    full_probabilities = np.asarray(full_probabilities, dtype=float)
    retrained_probabilities = np.asarray(retrained_probabilities, dtype=float)

    def compute_metrics(
        y: np.ndarray,
        full_p: np.ndarray,
        reduced_p: np.ndarray,
    ) -> Dict[str, float]:
        full_auroc = roc_auc_score(y, full_p)
        reduced_auroc = roc_auc_score(y, reduced_p)
        full_auprc = average_precision_score(y, full_p)
        reduced_auprc = average_precision_score(y, reduced_p)
        full_brier = brier_score_loss(y, full_p)
        reduced_brier = brier_score_loss(y, reduced_p)
        full_oe_error = _calibration_absolute_oe_error(y, full_p)
        reduced_oe_error = _calibration_absolute_oe_error(y, reduced_p)

        return {
            "AUROC": full_auroc - reduced_auroc,
            "AUPRC": full_auprc - reduced_auprc,
            "Brier": reduced_brier - full_brier,
            "Absolute_OE_error": reduced_oe_error - full_oe_error,
            "full_AUROC": full_auroc,
            "retrained_AUROC": reduced_auroc,
            "full_AUPRC": full_auprc,
            "retrained_AUPRC": reduced_auprc,
            "full_Brier": full_brier,
            "retrained_Brier": reduced_brier,
            "full_Absolute_OE_error": full_oe_error,
            "retrained_Absolute_OE_error": reduced_oe_error,
        }

    point = compute_metrics(
        y_true,
        full_probabilities,
        retrained_probabilities,
    )

    rng = np.random.default_rng(random_state)
    bootstrap_values: Dict[str, List[float]] = {
        "AUROC": [],
        "AUPRC": [],
        "Brier": [],
        "Absolute_OE_error": [],
    }

    for _ in range(n_bootstrap):
        indices = stratified_bootstrap_indices(y_true, rng)
        bootstrap_metrics = compute_metrics(
            y_true[indices],
            full_probabilities[indices],
            retrained_probabilities[indices],
        )
        for metric in bootstrap_values:
            bootstrap_values[metric].append(
                float(bootstrap_metrics[metric])
            )

    rows: List[Dict[str, Any]] = []

    for metric, values in bootstrap_values.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]

        if len(array):
            ci_low = float(np.quantile(array, 0.025))
            ci_high = float(np.quantile(array, 0.975))
            p_two_sided = min(
                1.0,
                2.0
                * min(
                    float(np.mean(array <= 0.0)),
                    float(np.mean(array >= 0.0)),
                ),
            )
        else:
            ci_low = np.nan
            ci_high = np.nan
            p_two_sided = np.nan

        rows.append(
            {
                "analysis": analysis,
                "removed_feature_set": removed_feature_set,
                "model": model_name,
                "metric": metric,
                "full_estimate": float(point[f"full_{metric}"]),
                "retrained_estimate": float(point[f"retrained_{metric}"]),
                "difference_favoring_complete_model": float(point[metric]),
                "CI_low": ci_low,
                "CI_high": ci_high,
                "bootstrap_p_two_sided": p_two_sided,
                "n_bootstrap": int(n_bootstrap),
            }
        )

    return pd.DataFrame(rows)


def fit_retrained_feature_set_variant(
    *,
    variant_name: str,
    analysis_type: str,
    excluded_raw_columns: Sequence[str],
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fold_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: PipelineConfig,
    output_directory: pathlib.Path,
    retrain_hybrid: bool,
) -> Dict[str, Any]:
    """Refit the complete DARN and optional XGBoost/blend pipeline.

    Every fold refits:
      preprocessing -> variance filtering -> supervised feature selection ->
      DARN -> XGBoost -> fold-specific cross-fitted calibration.

    The blend weight, blend calibration, and operating threshold are then
    selected using development OOF predictions only.
    """
    variant_safe_name = _safe_variant_name(variant_name)
    variant_directory = output_directory / variant_safe_name
    variant_directory.mkdir(parents=True, exist_ok=True)

    excluded_raw_columns = [
        str(column)
        for column in excluded_raw_columns
        if str(column) in X_development.columns
    ]

    if not excluded_raw_columns:
        raise ValueError(
            f"{variant_name}: no requested raw columns were present."
        )

    X_development_variant = X_development.drop(
        columns=excluded_raw_columns
    ).copy()
    X_test_variant = X_test.drop(
        columns=excluded_raw_columns
    ).copy()

    if X_development_variant.shape[1] == 0:
        raise ValueError(
            f"{variant_name}: removing the requested feature set left no predictors."
        )

    np.save(
        variant_directory / "excluded_raw_columns.npy",
        np.asarray(excluded_raw_columns, dtype=object),
    )

    darn_oof = np.full(
        len(y_development),
        np.nan,
        dtype=float,
    )
    darn_test_folds: List[np.ndarray] = []

    xgb_enabled = bool(
        retrain_hybrid
        and config.run_xgboost
        and XGBOOST_AVAILABLE
    )
    xgb_oof = (
        np.full(len(y_development), np.nan, dtype=float)
        if xgb_enabled
        else None
    )
    xgb_test_folds: List[np.ndarray] = []

    fold_rows: List[Dict[str, Any]] = []
    selection_rows: List[pd.DataFrame] = []

    print("\n" + "=" * 100)
    print(
        f"TRUE RETRAINING: {analysis_type} | {variant_name}"
    )
    print("=" * 100)
    print(
        f"Excluded raw columns: {len(excluded_raw_columns):,}"
    )
    print(
        f"Remaining raw predictors: {X_development_variant.shape[1]:,}"
    )
    print(
        "Hybrid retraining:",
        "enabled" if xgb_enabled else "disabled",
    )

    for fold_number, (
        fold_train_indices,
        fold_validation_indices,
    ) in enumerate(
        fold_splits,
        start=1,
    ):
        fold_seed = config.random_state + fold_number * 100
        fold_directory = (
            variant_directory
            / f"fold_{fold_number}"
        )
        fold_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        X_fold_train_raw = X_development_variant.iloc[
            fold_train_indices
        ].copy()
        y_fold_train = y_development[
            fold_train_indices
        ]

        X_fold_validation_raw = X_development_variant.iloc[
            fold_validation_indices
        ].copy()
        y_fold_validation = y_development[
            fold_validation_indices
        ]

        preprocessor = make_preprocessor(
            X_fold_train_raw
        )

        X_fold_train = preprocessor.fit_transform(
            X_fold_train_raw
        )
        X_fold_validation = preprocessor.transform(
            X_fold_validation_raw
        )
        X_fold_test = preprocessor.transform(
            X_test_variant
        )

        X_fold_train = np.nan_to_num(
            X_fold_train,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        X_fold_validation = np.nan_to_num(
            X_fold_validation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        X_fold_test = np.nan_to_num(
            X_fold_test,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        variance_selector = VarianceThreshold(
            threshold=config.variance_threshold
        )

        X_fold_train = variance_selector.fit_transform(
            X_fold_train
        )
        X_fold_validation = variance_selector.transform(
            X_fold_validation
        )
        X_fold_test = variance_selector.transform(
            X_fold_test
        )

        (
            domain_names,
            domain_indices,
            retained_feature_names,
        ) = build_domain_indices(
            preprocessor,
            variance_selector,
            raw_feature_columns=X_fold_train_raw.columns.tolist(),
        )

        (
            selected_indices,
            selected_domain_names,
            selected_domain_slices,
            selected_domain_indices,
            selected_feature_names,
            selection_summary,
            feature_score_table,
        ) = select_features_by_domain(
            X_fold_train,
            y_fold_train,
            domain_names,
            domain_indices,
            retained_feature_names,
            config.domain_caps(),
        )

        if len(selected_domain_names) < 2:
            raise RuntimeError(
                f"{variant_name}, fold {fold_number}: fewer than two domains "
                "remained after feature selection."
            )

        X_fold_train = X_fold_train[
            :,
            selected_indices,
        ]
        X_fold_validation = X_fold_validation[
            :,
            selected_indices,
        ]
        X_fold_test = X_fold_test[
            :,
            selected_indices,
        ]

        selection_summary = selection_summary.copy()
        selection_summary.insert(
            0,
            "fold",
            fold_number,
        )
        selection_summary.insert(
            0,
            "variant",
            variant_name,
        )
        selection_rows.append(
            selection_summary
        )

        if config.lodo_save_models:
            joblib.dump(
                preprocessor,
                fold_directory
                / "preprocessor.joblib",
            )
            joblib.dump(
                variance_selector,
                fold_directory
                / "variance_selector.joblib",
            )
            np.save(
                fold_directory
                / "selected_original_indices.npy",
                selected_indices,
            )
            np.save(
                fold_directory
                / "selected_feature_names.npy",
                selected_feature_names,
            )

        # --------------------------------------------------------------
        # Uniform DARN retraining
        # --------------------------------------------------------------
        darn_validation_seed_predictions: List[np.ndarray] = []
        darn_test_seed_predictions: List[np.ndarray] = []
        darn_best_rows: List[Dict[str, float]] = []

        for seed_offset in range(
            config.darn_seeds_per_fold
        ):
            seed = fold_seed + seed_offset

            darn_model = UniformDARNClassifier(
                domain_slices=selected_domain_slices,
                hidden_dim=config.darn_hidden_dim,
                dropout=config.darn_dropout,
                feature_dropout=config.darn_feature_dropout,
            )

            (
                darn_model,
                darn_history,
                darn_best,
            ) = train_torch_binary_model(
                darn_model,
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
                learning_rate=config.darn_learning_rate,
                weight_decay=config.darn_weight_decay,
                batch_size=config.darn_batch_size,
                max_epochs=config.darn_max_epochs,
                patience=config.darn_patience,
                pos_weight=config.darn_pos_weight,
                focal_gamma=config.darn_focal_gamma,
                seed=seed,
                label=(
                    f"LODO {variant_name} DARN "
                    f"F{fold_number} S{seed}"
                ),
            )

            device = next(
                darn_model.parameters()
            ).device

            validation_probability = (
                predict_torch_model(
                    darn_model,
                    X_fold_validation,
                    config.prediction_batch_size,
                    device,
                )
            )

            test_probability = (
                predict_torch_model(
                    darn_model,
                    X_fold_test,
                    config.prediction_batch_size,
                    device,
                )
            )

            darn_validation_seed_predictions.append(
                validation_probability
            )
            darn_test_seed_predictions.append(
                test_probability
            )
            darn_best_rows.append(
                darn_best
            )

            if config.lodo_save_models:
                torch.save(
                    darn_model.state_dict(),
                    fold_directory
                    / f"uniform_darn_seed_{seed}.pt",
                )

            darn_history.to_csv(
                fold_directory
                / f"uniform_darn_history_seed_{seed}.csv",
                index=False,
            )

            darn_model.to("cpu")
            del darn_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        darn_validation_raw = np.mean(
            np.vstack(
                darn_validation_seed_predictions
            ),
            axis=0,
        )

        darn_test_raw = np.mean(
            np.vstack(
                darn_test_seed_predictions
            ),
            axis=0,
        )

        (
            darn_validation_calibrated,
            darn_test_calibrated,
            darn_calibrator,
        ) = cross_fitted_fold_calibration(
            y_fold_validation,
            darn_validation_raw,
            darn_test_raw,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=fold_seed + 4011,
        )

        darn_oof[
            fold_validation_indices
        ] = darn_validation_calibrated

        darn_test_folds.append(
            darn_test_calibrated
        )

        if config.lodo_save_models:
            joblib.dump(
                darn_calibrator,
                fold_directory
                / "calibrator_uniform_darn.joblib",
            )

        fold_rows.append(
            {
                "analysis": analysis_type,
                "variant": variant_name,
                "fold": fold_number,
                "model": "Uniform DARN",
                "validation_AUROC": float(
                    roc_auc_score(
                        y_fold_validation,
                        darn_validation_calibrated,
                    )
                ),
                "validation_AUPRC": float(
                    average_precision_score(
                        y_fold_validation,
                        darn_validation_calibrated,
                    )
                ),
                "selected_features": int(
                    len(selected_feature_names)
                ),
                "selected_domains": int(
                    len(selected_domain_names)
                ),
                "best_epoch_mean": float(
                    np.mean(
                        [
                            row[
                                "best_epoch"
                            ]
                            for row in darn_best_rows
                        ]
                    )
                ),
            }
        )

        # --------------------------------------------------------------
        # XGBoost retraining for the final hybrid
        # --------------------------------------------------------------
        if xgb_enabled:
            xgb_model = make_xgboost_model(
                y_fold_train,
                config,
                seed=fold_seed,
            )

            xgb_model = fit_xgboost_with_early_stopping(
                xgb_model,
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
                config.xgb_early_stopping_rounds,
            )

            xgb_validation_raw = (
                xgb_model.predict_proba(
                    X_fold_validation
                )[:, 1]
            )

            xgb_test_raw = (
                xgb_model.predict_proba(
                    X_fold_test
                )[:, 1]
            )

            (
                xgb_validation_calibrated,
                xgb_test_calibrated,
                xgb_calibrator,
            ) = cross_fitted_fold_calibration(
                y_fold_validation,
                xgb_validation_raw,
                xgb_test_raw,
                n_splits=config.platt_crossfit_folds,
                c_value=config.platt_c,
                random_state=fold_seed + 4012,
            )

            assert xgb_oof is not None

            xgb_oof[
                fold_validation_indices
            ] = xgb_validation_calibrated

            xgb_test_folds.append(
                xgb_test_calibrated
            )

            if config.lodo_save_models:
                joblib.dump(
                    xgb_model,
                    fold_directory
                    / "xgboost.joblib",
                )
                joblib.dump(
                    xgb_calibrator,
                    fold_directory
                    / "calibrator_xgboost.joblib",
                )

            fold_rows.append(
                {
                    "analysis": analysis_type,
                    "variant": variant_name,
                    "fold": fold_number,
                    "model": "XGBoost",
                    "validation_AUROC": float(
                        roc_auc_score(
                            y_fold_validation,
                            xgb_validation_calibrated,
                        )
                    ),
                    "validation_AUPRC": float(
                        average_precision_score(
                            y_fold_validation,
                            xgb_validation_calibrated,
                        )
                    ),
                    "selected_features": int(
                        len(selected_feature_names)
                    ),
                    "selected_domains": int(
                        len(selected_domain_names)
                    ),
                    "best_iteration": float(
                        getattr(
                            xgb_model,
                            "best_iteration",
                            np.nan,
                        )
                    ),
                }
            )

            del xgb_model

    if not np.isfinite(darn_oof).all():
        raise RuntimeError(
            f"{variant_name}: missing DARN OOF probabilities."
        )

    darn_test = np.mean(
        np.vstack(
            darn_test_folds
        ),
        axis=0,
    )

    darn_development_threshold = (
        select_operating_threshold(
            y_development,
            darn_oof,
            strategy=config.threshold_strategy,
            target_sensitivity=config.target_sensitivity,
            target_specificity=config.selective_target_specificity,
        )
    )

    (
        darn_test_threshold,
        darn_transport,
    ) = transport_threshold_to_test(
        darn_oof,
        darn_development_threshold,
        darn_test,
        mode=config.threshold_transport_mode,
        max_rate_ratio=config.threshold_transport_rate_ratio,
        max_absolute_gap=config.threshold_transport_absolute_gap,
        degenerate_rate=config.threshold_transport_degenerate_rate,
    )

    variant_probabilities: Dict[
        str,
        np.ndarray,
    ] = {
        "Uniform DARN": darn_test,
    }

    variant_oof_probabilities: Dict[
        str,
        np.ndarray,
    ] = {
        "Uniform DARN": darn_oof,
    }

    variant_thresholds: Dict[
        str,
        float,
    ] = {
        "Uniform DARN": darn_test_threshold,
    }

    metadata: Dict[
        str,
        Any,
    ] = {
        "analysis": analysis_type,
        "variant": variant_name,
        "excluded_raw_columns": excluded_raw_columns,
        "remaining_raw_predictors": int(
            X_development_variant.shape[1]
        ),
        "darn_threshold_transport": darn_transport,
        "hybrid_retrained": bool(xgb_enabled),
    }

    if xgb_enabled:
        assert xgb_oof is not None

        if not np.isfinite(
            xgb_oof
        ).all():
            raise RuntimeError(
                f"{variant_name}: missing XGBoost OOF probabilities."
            )

        xgb_test = np.mean(
            np.vstack(
                xgb_test_folds
            ),
            axis=0,
        )

        best_darn_weight, weight_search = (
            optimize_darn_xgb_blend(
                y_development,
                darn_oof,
                xgb_oof,
                grid_size=config.hybrid_blend_grid_size,
                min_darn_weight=config.hybrid_min_darn_weight,
                max_darn_weight=config.hybrid_max_darn_weight,
            )
        )

        blend_oof_raw = (
            best_darn_weight
            * darn_oof
            + (
                1.0
                - best_darn_weight
            )
            * xgb_oof
        )

        blend_test_raw = (
            best_darn_weight
            * darn_test
            + (
                1.0
                - best_darn_weight
            )
            * xgb_test
        )

        (
            blend_oof,
            blend_test,
            blend_calibrator,
        ) = cross_fitted_fold_calibration(
            y_development,
            blend_oof_raw,
            blend_test_raw,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=config.random_state + 4801,
        )

        blend_development_threshold = (
            select_operating_threshold(
                y_development,
                blend_oof,
                strategy=config.threshold_strategy,
                target_sensitivity=config.target_sensitivity,
                target_specificity=config.selective_target_specificity,
            )
        )

        (
            blend_test_threshold,
            blend_transport,
        ) = transport_threshold_to_test(
            blend_oof,
            blend_development_threshold,
            blend_test,
            mode=config.threshold_transport_mode,
            max_rate_ratio=config.threshold_transport_rate_ratio,
            max_absolute_gap=config.threshold_transport_absolute_gap,
            degenerate_rate=config.threshold_transport_degenerate_rate,
        )

        variant_probabilities[
            "Hybrid DARN-XGB Blend"
        ] = blend_test

        variant_oof_probabilities[
            "Hybrid DARN-XGB Blend"
        ] = blend_oof

        variant_thresholds[
            "Hybrid DARN-XGB Blend"
        ] = blend_test_threshold

        weight_search.to_csv(
            variant_directory
            / "hybrid_blend_weight_search.csv",
            index=False,
        )

        if config.lodo_save_models:
            joblib.dump(
                blend_calibrator,
                variant_directory
                / "calibrator_hybrid_darn_xgb_blend.joblib",
            )

        metadata.update(
            {
                "hybrid_darn_weight": float(
                    best_darn_weight
                ),
                "hybrid_xgboost_weight": float(
                    1.0
                    - best_darn_weight
                ),
                "hybrid_threshold_transport": blend_transport,
            }
        )

    metric_rows: List[
        Dict[str, Any]
    ] = []

    prediction_table = pd.DataFrame(
        {
            "y_true": y_test,
        }
    )

    oof_table = pd.DataFrame(
        {
            "y_true": y_development,
        }
    )

    for model_name, probabilities in (
        variant_probabilities.items()
    ):
        metrics = calculate_binary_metrics(
            y_test,
            probabilities,
            variant_thresholds[
                model_name
            ],
        )

        metric_rows.append(
            {
                "analysis": analysis_type,
                "variant": variant_name,
                "model": model_name,
                **metrics,
            }
        )

        safe_model = _safe_variant_name(
            model_name
        )

        prediction_table[
            f"{safe_model}_calibrated_probability"
        ] = probabilities

        oof_table[
            f"{safe_model}_calibrated_probability"
        ] = variant_oof_probabilities[
            model_name
        ]

    fold_performance = pd.DataFrame(
        fold_rows
    )

    fold_performance.to_csv(
        variant_directory
        / "fold_validation_performance.csv",
        index=False,
    )

    if selection_rows:
        pd.concat(
            selection_rows,
            ignore_index=True,
        ).to_csv(
            variant_directory
            / "domain_selection_summary_all_folds.csv",
            index=False,
        )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    metrics_table.to_csv(
        variant_directory
        / "test_metrics.csv",
        index=False,
    )

    prediction_table.to_csv(
        variant_directory
        / "test_predictions.csv",
        index=False,
    )

    oof_table.to_csv(
        variant_directory
        / "development_oof_predictions.csv",
        index=False,
    )

    with open(
        variant_directory
        / "variant_metadata.json",
        "w",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=json_default,
        )

    return {
        "analysis": analysis_type,
        "variant": variant_name,
        "excluded_raw_columns": excluded_raw_columns,
        "probabilities": variant_probabilities,
        "oof_probabilities": variant_oof_probabilities,
        "thresholds": variant_thresholds,
        "metrics": metrics_table,
        "fold_performance": fold_performance,
        "metadata": metadata,
        "output_directory": variant_directory,
    }


def save_leave_one_domain_out_figure(
    comparison_table: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    if comparison_table.empty:
        return

    models = [
        model
        for model in (
            "Uniform DARN",
            "Hybrid DARN-XGB Blend",
        )
        if model
        in comparison_table[
            "model"
        ].unique()
    ]

    figure, axes = plt.subplots(
        2,
        max(
            len(models),
            1,
        ),
        figsize=(
            8.2
            * max(
                len(models),
                1,
            ),
            11.5,
        ),
        squeeze=False,
    )

    cmap = plt.get_cmap(
        "plasma"
    )

    for column_number, model_name in enumerate(
        models
    ):
        model_table = comparison_table.loc[
            comparison_table[
                "model"
            ].eq(
                model_name
            )
        ].copy()

        domain_order = (
            model_table.loc[
                model_table[
                    "metric"
                ].eq(
                    "AUPRC"
                )
            ]
            .sort_values(
                "difference_favoring_complete_model",
                ascending=True,
            )[
                "removed_feature_set"
            ]
            .tolist()
        )

        if not domain_order:
            domain_order = sorted(
                model_table[
                    "removed_feature_set"
                ].unique()
            )

        colors = {
            domain: cmap(
                value
            )
            for domain, value in zip(
                domain_order,
                np.linspace(
                    0.10,
                    0.90,
                    max(
                        len(
                            domain_order
                        ),
                        1,
                    ),
                ),
            )
        }

        for row_number, (
            metrics,
            title,
            xlabel,
        ) in enumerate(
            [
                (
                    [
                        "AUPRC",
                        "AUROC",
                    ],
                    (
                        f"{chr(65 + column_number)}. "
                        f"{model_name}: discrimination"
                    ),
                    (
                        "Complete model − domain-removed model\n"
                        "(positive values favor the complete model)"
                    ),
                ),
                (
                    [
                        "Brier",
                        "Absolute_OE_error",
                    ],
                    (
                        f"{chr(67 + column_number)}. "
                        f"{model_name}: calibration"
                    ),
                    (
                        "Domain-removed error − complete-model error\n"
                        "(positive values favor the complete model)"
                    ),
                ),
            ]
        ):
            axis = axes[
                row_number,
                column_number,
            ]

            y_positions = np.arange(
                len(
                    domain_order
                )
            )
            offsets = np.linspace(
                -0.15,
                0.15,
                len(
                    metrics
                ),
            )

            for metric_offset, (
                metric,
                offset,
            ) in enumerate(
                zip(
                    metrics,
                    offsets,
                )
            ):
                metric_table = (
                    model_table.loc[
                        model_table[
                            "metric"
                        ].eq(
                            metric
                        )
                    ]
                    .set_index(
                        "removed_feature_set"
                    )
                    .reindex(
                        domain_order
                    )
                )

                estimates = metric_table[
                    "difference_favoring_complete_model"
                ].to_numpy(
                    dtype=float
                )

                lower = metric_table[
                    "CI_low"
                ].to_numpy(
                    dtype=float
                )

                upper = metric_table[
                    "CI_high"
                ].to_numpy(
                    dtype=float
                )

                x_error = np.vstack(
                    [
                        np.maximum(
                            0.0,
                            estimates
                            - lower,
                        ),
                        np.maximum(
                            0.0,
                            upper
                            - estimates,
                        ),
                    ]
                )

                axis.errorbar(
                    estimates,
                    y_positions
                    + offset,
                    xerr=x_error,
                    fmt=(
                        "o"
                        if metric_offset
                        == 0
                        else "s"
                    ),
                    markersize=6,
                    capsize=2.5,
                    linewidth=1.25,
                    label=metric.replace(
                        "_",
                        " ",
                    ),
                    color=(
                        cmap(
                            0.20
                            + 0.55
                            * metric_offset
                        )
                    ),
                )

            axis.axvline(
                0.0,
                color="gray",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_yticks(
                y_positions
            )
            axis.set_yticklabels(
                [
                    domain.replace(
                        "_",
                        " "
                    )
                    for domain in domain_order
                ]
            )
            axis.set_xlabel(
                xlabel
            )
            axis.set_title(
                title,
                loc="left",
                fontweight="bold",
            )
            axis.grid(
                axis="x",
                linestyle="--",
                alpha=0.20,
            )
            axis.legend(
                frameon=False,
                fontsize=8.5,
            )

    figure.suptitle(
        "Performance changes after true leave-one-domain-out retraining",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    figure.text(
        0.04,
        0.015,
        (
            "Each analysis refit preprocessing, variance filtering, supervised "
            "feature selection, five fold-specific models, calibration, and "
            "the development-defined threshold. Confidence intervals are "
            "paired stratified-bootstrap intervals on the unchanged holdout."
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.045,
            0.99,
            0.95,
        ),
        h_pad=2.8,
        w_pad=2.4,
    )

    figure.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )
    figure.savefig(
        output_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )
    figure.savefig(
        output_path.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def run_leave_one_domain_out_retraining(
    *,
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    fold_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    full_test_probabilities: Dict[str, np.ndarray],
    config: PipelineConfig,
    run_dir: pathlib.Path,
) -> Dict[str, Any]:
    output_directory = (
        run_dir
        / "leave_one_domain_out_retraining"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_groups = get_feature_groups(
        X_development.columns
    )

    domain_order = [
        domain
        for domain in (
            "Demographics",
            "Diagnoses",
            "Durations",
            "Interactions",
            "Intraop_Vitals",
            "Medications",
            "Preop_Labs",
            "Procedures",
        )
        if domain in raw_groups
    ]

    variant_results: List[
        Dict[str, Any]
    ] = []
    comparison_tables: List[
        pd.DataFrame
    ] = []
    all_metric_tables: List[
        pd.DataFrame
    ] = []

    prediction_table = pd.DataFrame(
        {
            "y_true": y_test,
        }
    )

    for domain_number, domain in enumerate(
        domain_order,
        start=1,
    ):
        excluded_columns = (
            raw_columns_removed_for_domain(
                X_development.columns,
                domain,
            )
        )

        result = (
            fit_retrained_feature_set_variant(
                variant_name=domain,
                analysis_type=(
                    "True leave-one-domain-out retraining"
                ),
                excluded_raw_columns=excluded_columns,
                X_development=X_development,
                y_development=y_development,
                X_test=X_test,
                y_test=y_test,
                fold_splits=fold_splits,
                config=config,
                output_directory=output_directory,
                retrain_hybrid=config.lodo_retrain_hybrid,
            )
        )

        variant_results.append(
            result
        )
        all_metric_tables.append(
            result[
                "metrics"
            ]
        )

        for model_name, reduced_probability in (
            result[
                "probabilities"
            ].items()
        ):
            if model_name not in full_test_probabilities:
                continue

            comparison = (
                paired_bootstrap_retraining_difference(
                    y_test,
                    full_test_probabilities[
                        model_name
                    ],
                    reduced_probability,
                    analysis=(
                        "True leave-one-domain-out retraining"
                    ),
                    removed_feature_set=domain,
                    model_name=model_name,
                    n_bootstrap=config.lodo_bootstrap,
                    random_state=(
                        config.random_state
                        + 6200
                        + 101
                        * domain_number
                    ),
                )
            )

            comparison_tables.append(
                comparison
            )

            prediction_table[
                (
                    f"{_safe_variant_name(model_name)}"
                    f"_without_{_safe_variant_name(domain)}"
                )
            ] = reduced_probability

    metrics_table = (
        pd.concat(
            all_metric_tables,
            ignore_index=True,
        )
        if all_metric_tables
        else pd.DataFrame()
    )

    comparison_table = (
        pd.concat(
            comparison_tables,
            ignore_index=True,
        )
        if comparison_tables
        else pd.DataFrame()
    )

    metrics_table.to_csv(
        output_directory
        / "leave_one_domain_out_retrained_test_metrics.csv",
        index=False,
    )

    comparison_table.to_csv(
        output_directory
        / "leave_one_domain_out_paired_bootstrap_differences.csv",
        index=False,
    )

    prediction_table.to_csv(
        output_directory
        / "leave_one_domain_out_test_probabilities.csv",
        index=False,
    )

    figure_path = (
        output_directory
        / "FigS_Leave_One_Domain_Out_Retraining.png"
    )

    save_leave_one_domain_out_figure(
        comparison_table,
        figure_path,
    )

    return {
        "domains": domain_order,
        "variant_results": variant_results,
        "metrics": metrics_table,
        "paired_bootstrap": comparison_table,
        "predictions": prediction_table,
        "figure": figure_path,
        "output_directory": output_directory,
    }


def identify_eye_ear_diagnosis_column(
    frame: pd.DataFrame,
) -> Optional[str]:
    candidates = [
        column
        for column in frame.columns
        if str(column).upper().startswith(
            "DIAG_"
        )
        and "eye" in re.sub(
            r"[^a-z0-9]+",
            "",
            str(column).lower(),
        )
        and "ear" in re.sub(
            r"[^a-z0-9]+",
            "",
            str(column).lower(),
        )
    ]
    return candidates[0] if candidates else None


def eye_ear_adjusted_association_analysis(
    df: pd.DataFrame,
    y: np.ndarray,
    eye_ear_column: str,
    output_path: pathlib.Path,
) -> pd.DataFrame:
    """Estimate staged adjusted eye/ear mortality associations when possible."""
    eye_ear = (
        pd.to_numeric(
            df[
                eye_ear_column
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for status in (
        0,
        1,
    ):
        mask = eye_ear.eq(
            status
        ).to_numpy()

        n = int(
            mask.sum()
        )
        events = int(
            np.asarray(
                y
            )[
                mask
            ].sum()
        )

        rows.append(
            {
                "analysis": "Descriptive mortality",
                "model_stage": (
                    "Eye/ear absent"
                    if status == 0
                    else "Eye/ear present"
                ),
                "N": n,
                "Deaths": events,
                "Mortality_rate": (
                    events / n
                    if n
                    else np.nan
                ),
                "odds_ratio": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "p_value": np.nan,
            }
        )

    if not STATSMODELS_AVAILABLE:
        pd.DataFrame(rows).to_csv(
            output_path,
            index=False,
        )
        print(
            "statsmodels unavailable; saved descriptive eye/ear results only."
        )
        return pd.DataFrame(
            rows
        )

    department_column = next(
        (
            column
            for column in (
                "department",
                "surgical_department",
                "surgery_department",
                "department_name",
                "op_department",
                "optype",
                "surgery_type",
                "surgical_service",
            )
            if column in df.columns
        ),
        None,
    )

    base = pd.DataFrame(
        {
            "eye_ear": eye_ear.astype(
                float
            ),
            "age": pd.to_numeric(
                df.get(
                    "age"
                ),
                errors="coerce",
            ),
            "asa": (
                parse_asa_numeric(
                    df[
                        "asa"
                    ]
                )
                if "asa" in df.columns
                else np.nan
            ),
            "emergency": pd.to_numeric(
                df.get(
                    "emop"
                ),
                errors="coerce",
            ),
        }
    )

    if "sex" in df.columns:
        base[
            "sex"
        ] = (
            df[
                "sex"
            ]
            .astype(
                "string"
            )
            .fillna(
                "Missing"
            )
        )

    if department_column is not None:
        department = (
            df[
                department_column
            ]
            .astype(
                "string"
            )
            .fillna(
                "Missing"
            )
        )

        common = (
            department.value_counts()
            .head(
                12
            )
            .index
        )

        base[
            "department"
        ] = department.where(
            department.isin(
                common
            ),
            "Other",
        )

    stages: List[
        Tuple[
            str,
            List[str],
        ]
    ] = [
        (
            "Unadjusted",
            [
                "eye_ear",
            ],
        ),
        (
            "Age and sex adjusted",
            [
                "eye_ear",
                "age",
                "sex",
            ],
        ),
        (
            "Age, sex, ASA, and urgency adjusted",
            [
                "eye_ear",
                "age",
                "sex",
                "asa",
                "emergency",
            ],
        ),
    ]

    if department_column is not None:
        stages.append(
            (
                "Additionally adjusted for surgical department",
                [
                    "eye_ear",
                    "age",
                    "sex",
                    "asa",
                    "emergency",
                    "department",
                ],
            )
        )

    for stage_name, columns in stages:
        available_columns = [
            column
            for column in columns
            if column in base.columns
        ]

        design = base.loc[
            :,
            available_columns,
        ].copy()

        numeric_columns = design.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:
            design[
                column
            ] = design[
                column
            ].fillna(
                design[
                    column
                ].median()
            )

        categorical_columns = [
            column
            for column in design.columns
            if column not in numeric_columns
        ]

        for column in categorical_columns:
            design[
                column
            ] = (
                design[
                    column
                ]
                .astype(
                    "string"
                )
                .fillna(
                    "Missing"
                )
            )

        design = pd.get_dummies(
            design,
            columns=categorical_columns,
            drop_first=True,
            dtype=float,
        )

        design = sm.add_constant(
            design.astype(
                float
            ),
            has_constant="add",
        )

        try:
            fitted = sm.GLM(
                np.asarray(
                    y,
                    dtype=float,
                ),
                design,
                family=sm.families.Binomial(),
            ).fit(
                cov_type="HC0"
            )

            coefficient = float(
                fitted.params[
                    "eye_ear"
                ]
            )
            standard_error = float(
                fitted.bse[
                    "eye_ear"
                ]
            )

            rows.append(
                {
                    "analysis": "Adjusted odds ratio",
                    "model_stage": stage_name,
                    "N": int(
                        len(
                            design
                        )
                    ),
                    "Deaths": int(
                        np.asarray(
                            y
                        ).sum()
                    ),
                    "Mortality_rate": float(
                        np.asarray(
                            y
                        ).mean()
                    ),
                    "odds_ratio": float(
                        np.exp(
                            coefficient
                        )
                    ),
                    "CI_low": float(
                        np.exp(
                            coefficient
                            - 1.959963984540054
                            * standard_error
                        )
                    ),
                    "CI_high": float(
                        np.exp(
                            coefficient
                            + 1.959963984540054
                            * standard_error
                        )
                    ),
                    "p_value": float(
                        fitted.pvalues[
                            "eye_ear"
                        ]
                    ),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "analysis": "Adjusted odds ratio",
                    "model_stage": stage_name,
                    "N": int(
                        len(
                            design
                        )
                    ),
                    "Deaths": int(
                        np.asarray(
                            y
                        ).sum()
                    ),
                    "Mortality_rate": float(
                        np.asarray(
                            y
                        ).mean()
                    ),
                    "odds_ratio": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p_value": np.nan,
                    "note": str(
                        exc
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )
    results.to_csv(
        output_path,
        index=False,
    )
    return results


def build_prespecified_tail_calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Clinically focused risk strata rather than ten nearly empty deciles."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    percentile = (
        pd.Series(
            probabilities
        )
        .rank(
            method="first",
            pct=True,
        )
        .to_numpy()
    )

    definitions = [
        (
            "Bottom 90%",
            0.00,
            0.90,
        ),
        (
            "90th-95th percentile",
            0.90,
            0.95,
        ),
        (
            "95th-98th percentile",
            0.95,
            0.98,
        ),
        (
            "98th-99th percentile",
            0.98,
            0.99,
        ),
        (
            "Top 1%",
            0.99,
            1.01,
        ),
    ]

    rows: List[
        Dict[str, Any]
    ] = []

    total_events = int(
        y_true.sum()
    )

    for order, (
        label,
        lower,
        upper,
    ) in enumerate(
        definitions,
        start=1,
    ):
        mask = (
            percentile > lower
        ) & (
            percentile <= upper
        )

        n = int(
            mask.sum()
        )
        events = int(
            y_true[
                mask
            ].sum()
        )

        if n == 0:
            continue

        rows.append(
            {
                "risk_stratum_order": order,
                "risk_stratum": label,
                "N": n,
                "Deaths": events,
                "mean_predicted_probability": float(
                    probabilities[
                        mask
                    ].mean()
                ),
                "median_predicted_probability": float(
                    np.median(
                        probabilities[
                            mask
                        ]
                    )
                ),
                "observed_mortality": float(
                    y_true[
                        mask
                    ].mean()
                ),
                "death_capture_fraction": float(
                    events
                    / max(
                        total_events,
                        1,
                    )
                ),
                "PPV_within_stratum": float(
                    events
                    / n
                ),
                "number_needed_to_evaluate": (
                    float(
                        n
                        / events
                    )
                    if events
                    else np.inf
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_prespecified_tail_calibration_figure(
    table: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    if table.empty:
        return

    figure, (
        axis_a,
        axis_b,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            15.5,
            6.8,
        ),
    )

    cmap = plt.get_cmap(
        "plasma"
    )

    positions = np.arange(
        len(
            table
        )
    )

    colors = [
        cmap(
            value
        )
        for value in np.linspace(
            0.10,
            0.90,
            len(
                table
            ),
        )
    ]

    axis_a.plot(
        table[
            "mean_predicted_probability"
        ],
        table[
            "observed_mortality"
        ],
        marker="o",
        linewidth=2.0,
        color=cmap(
            0.55
        ),
    )

    upper = max(
        float(
            table[
                [
                    "mean_predicted_probability",
                    "observed_mortality",
                ]
            ].to_numpy().max()
        )
        * 1.15,
        1e-4,
    )

    axis_a.plot(
        [
            0.0,
            upper,
        ],
        [
            0.0,
            upper,
        ],
        linestyle=":",
        color="gray",
    )

    axis_a.set_xlim(
        0.0,
        upper,
    )
    axis_a.set_ylim(
        0.0,
        upper,
    )
    axis_a.set_xlabel(
        "Mean predicted mortality probability"
    )
    axis_a.set_ylabel(
        "Observed 30-day mortality"
    )
    axis_a.set_title(
        "A. Calibration across prespecified risk strata",
        loc="left",
        fontweight="bold",
    )
    axis_a.grid(
        linestyle="--",
        alpha=0.20,
    )

    bars = axis_b.bar(
        positions,
        100.0
        * table[
            "death_capture_fraction"
        ],
        color=colors,
        edgecolor="none",
    )

    axis_b.set_xticks(
        positions
    )
    axis_b.set_xticklabels(
        table[
            "risk_stratum"
        ],
        rotation=28,
        ha="right",
    )
    axis_b.set_ylabel(
        "Deaths captured (%)"
    )
    axis_b.set_xlabel(
        "Predicted-risk stratum"
    )
    axis_b.set_title(
        "B. Concentration of deaths across risk strata",
        loc="left",
        fontweight="bold",
    )
    axis_b.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    for bar, row in zip(
        bars,
        table.itertuples(),
    ):
        axis_b.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height()
            + 1.0,
            (
                f"{int(row.Deaths)}/{int(row.N)}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Tail-focused calibration and risk concentration",
        fontsize=16.5,
        fontweight="bold",
        y=0.985,
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.03,
            0.99,
            0.95,
        ),
        w_pad=2.8,
    )

    figure.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )
    figure.savefig(
        output_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )
    figure.savefig(
        output_path.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def create_false_negative_case_review(
    X_test_raw: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    false_negative_indices = np.where(
        (
            y_true == 1
        )
        & (
            probabilities
            < threshold
        )
    )[0]

    percentile = (
        pd.Series(
            probabilities
        )
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )

    preferred_columns = [
        column
        for column in (
            "age",
            "sex",
            "race",
            "asa",
            "bmi",
            "emop",
            "surgery_duration",
            "or_duration",
            "anesthesia_duration",
            "preop_fibrinogen",
            "preop_platelet",
            "preop_glucose",
            "intraop_mean_ebl",
            "intraop_max_ebl",
        )
        if column
        in X_test_raw.columns
    ]

    diagnosis_columns = [
        column
        for column in X_test_raw.columns
        if str(column).upper().startswith(
            "DIAG_"
        )
    ]

    rows: List[
        Dict[str, Any]
    ] = []

    for index in false_negative_indices:
        row: Dict[
            str,
            Any,
        ] = {
            "holdout_row_position": int(
                index
            ),
            "predicted_probability": float(
                probabilities[
                    index
                ]
            ),
            "development_defined_threshold": float(
                threshold
            ),
            "distance_below_threshold": float(
                threshold
                - probabilities[
                    index
                ]
            ),
            "predicted_risk_percentile": float(
                percentile[
                    index
                ]
            ),
            "raw_missing_count": int(
                X_test_raw.iloc[
                    index
                ].isna().sum()
            ),
            "raw_missing_fraction": float(
                X_test_raw.iloc[
                    index
                ].isna().mean()
            ),
        }

        for column in preferred_columns:
            row[
                column
            ] = X_test_raw.iloc[
                index
            ][
                column
            ]

        present_diagnoses: List[
            str
        ] = []

        for column in diagnosis_columns:
            value = pd.to_numeric(
                pd.Series(
                    [
                        X_test_raw.iloc[
                            index
                        ][
                            column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            if np.isfinite(
                value
            ) and value > 0:
                present_diagnoses.append(
                    str(
                        column
                    )
                )

        row[
            "present_diagnoses"
        ] = " | ".join(
            present_diagnoses[
                :25
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    overall_start = time.time()
    set_seed(config.random_state)
    run_dir = make_run_directory(config.out_root)

    print("\n" + "=" * 100)
    print("REVIEWER-COMPLETE UNIFORM DARN + DARN-XGBOOST HYBRIDS + BASELINES")
    print("=" * 100)
    print("Run directory :", run_dir)
    print("PyTorch       :", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
    print("XGBoost       :", "available" if XGBOOST_AVAILABLE else "not installed")

    with open(run_dir / "config.json", "w") as file:
        json.dump(asdict(config), file, indent=2, default=json_default)

    df_input = pd.read_csv(config.data_path, low_memory=False)
    df, cohort_flow = apply_reviewer_cohort_exclusions(
        df_input,
        target_column=config.target_column,
        exclude_asa6=config.exclude_asa6,
    )
    cohort_flow.to_csv(run_dir / "cohort_flow.csv", index=False)
    cohort_audit = cohort_audit_summary(df, config.target_column)
    cohort_audit.to_csv(run_dir / "cohort_audit_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("COHORT AUDIT")
    print("=" * 100)
    print(cohort_flow.to_string(index=False))
    if "emop" in df.columns:
        emergency_numeric = pd.to_numeric(df["emop"], errors="coerce")
        print(f"Emergency cases included: {int((emergency_numeric == 1).sum()):,}")

    if config.add_interactions:
        df = add_mortality_interactions(df)

    X_raw = extract_model_features_classification(df)
    y = df.loc[X_raw.index, config.target_column].astype(int).to_numpy()
    X_raw = clean_raw_predictors(X_raw)

    row_positions = np.arange(len(X_raw))
    source_row_positions = df.loc[X_raw.index, "_source_row_position"].to_numpy(dtype=int)

    development_positions, test_positions = train_test_split(
        row_positions,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    X_development = X_raw.iloc[development_positions].reset_index(drop=True)
    y_development = y[development_positions]
    X_test = X_raw.iloc[test_positions].reset_index(drop=True)
    y_test = y[test_positions]

    split_table = pd.DataFrame(
        {
            "final_cohort_row_position": row_positions,
            "source_row_position": source_row_positions,
            "split": np.where(
                np.isin(row_positions, test_positions), "test", "development"
            ),
            "outcome": y,
        }
    )
    split_table.to_csv(run_dir / "outer_split_rows.csv", index=False)

    print("\n" + "=" * 100)
    print("OUTER SPLIT")
    print("=" * 100)
    print(
        f"Development: N={len(y_development):,}, events={int(y_development.sum()):,}"
    )
    print(f"Test       : N={len(y_test):,}, events={int(y_test.sum()):,}")

    base_model_names = ["Uniform DARN", "ASA-only Logistic", "Logistic Regression", "MLP"]
    if config.run_xgboost and XGBOOST_AVAILABLE:
        base_model_names.append("XGBoost")

    model_names = list(base_model_names)

    oof_probabilities: Dict[str, np.ndarray] = {
        model_name: np.full(len(y_development), np.nan, dtype=float)
        for model_name in model_names
    }

    test_fold_probabilities: Dict[str, List[np.ndarray]] = {
        model_name: [] for model_name in model_names
    }

    # Fold-matched, cross-fitted calibration. Each development observation is
    # calibrated by a Platt model that did not use that observation, while each
    # fold model's test predictions are calibrated by a final calibrator fitted
    # only on that fold's held-out validation predictions.
    calibrated_oof_probabilities: Dict[str, np.ndarray] = {
        model_name: np.full(len(y_development), np.nan, dtype=float)
        for model_name in model_names
    }
    test_fold_calibrated_probabilities: Dict[str, List[np.ndarray]] = {
        model_name: [] for model_name in model_names
    }

    # DARN domain-ablation predictions are accumulated across folds.
    darn_ablated_test_probabilities: Dict[str, List[np.ndarray]] = {}
    fold_rows: List[Dict[str, Any]] = []
    gradient_shap_tables: List[pd.DataFrame] = []
    xgb_contribution_tables: List[pd.DataFrame] = []
    variance_filter_tables: List[pd.DataFrame] = []

    cross_validator = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    # Materialize the split indices once. The identical fold assignments are
    # reused for the complete model and every true retraining sensitivity
    # analysis, enabling paired and methodologically comparable results.
    fold_splits = list(
        cross_validator.split(
            X_development,
            y_development,
        )
    )

    for fold_number, (fold_train_indices, fold_validation_indices) in enumerate(
        fold_splits, start=1
    ):
        fold_start = time.time()
        fold_seed = config.random_state + fold_number * 100
        fold_dir = run_dir / "fold_artifacts" / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_fold_train_raw = X_development.iloc[fold_train_indices].copy()
        y_fold_train = y_development[fold_train_indices]
        X_fold_validation_raw = X_development.iloc[fold_validation_indices].copy()
        y_fold_validation = y_development[fold_validation_indices]

        print("\n" + "=" * 100)
        print(f"FOLD {fold_number}/{config.n_folds}")
        print("=" * 100)
        print(
            f"Train: N={len(y_fold_train):,}, events={int(y_fold_train.sum()):,} | "
            f"Validation: N={len(y_fold_validation):,}, events={int(y_fold_validation.sum()):,}"
        )

        preprocessor = make_preprocessor(X_fold_train_raw)
        X_fold_train = preprocessor.fit_transform(X_fold_train_raw)
        X_fold_validation = preprocessor.transform(X_fold_validation_raw)
        X_fold_test = preprocessor.transform(X_test)

        X_fold_train = np.nan_to_num(
            X_fold_train, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        X_fold_validation = np.nan_to_num(
            X_fold_validation, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        X_fold_test = np.nan_to_num(
            X_fold_test, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        variance_selector = VarianceThreshold(
            threshold=config.variance_threshold
        )
        all_transformed_feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
        X_fold_train = variance_selector.fit_transform(X_fold_train)
        X_fold_validation = variance_selector.transform(X_fold_validation)
        X_fold_test = variance_selector.transform(X_fold_test)

        variance_table = pd.DataFrame({
            "fold": fold_number,
            "feature": all_transformed_feature_names,
            "variance": variance_selector.variances_,
            "retained": variance_selector.get_support(),
        })
        variance_table.to_csv(fold_dir / "variance_filter_features.csv", index=False)
        variance_filter_tables.append(variance_table)

        domain_names, domain_indices, retained_feature_names = build_domain_indices(
            preprocessor,
            variance_selector,
            raw_feature_columns=X_fold_train_raw.columns.tolist(),
        )

        (
            selected_indices,
            selected_domain_names,
            selected_domain_slices,
            selected_domain_indices,
            selected_feature_names,
            selection_summary,
            feature_score_table,
        ) = select_features_by_domain(
            X_fold_train,
            y_fold_train,
            domain_names,
            domain_indices,
            retained_feature_names,
            config.domain_caps(),
        )

        X_fold_train = X_fold_train[:, selected_indices]
        X_fold_validation = X_fold_validation[:, selected_indices]
        X_fold_test = X_fold_test[:, selected_indices]

        if fold_number == 1:
            save_training_correlation_audit(
                X_fold_train,
                selected_feature_names,
                selected_domain_names,
                selected_domain_indices,
                feature_score_table,
                top_n=config.correlation_top_n,
                output_directory=run_dir / "correlation_audit",
            )

        print(f"Retained after variance filter : {len(retained_feature_names):,}")
        print(f"Selected for all models        : {len(selected_feature_names):,}")
        print(selection_summary.to_string(index=False))

        joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
        joblib.dump(variance_selector, fold_dir / "variance_selector.joblib")
        np.save(fold_dir / "selected_original_indices.npy", selected_indices)
        np.save(fold_dir / "selected_feature_names.npy", selected_feature_names)
        selection_summary.to_csv(fold_dir / "domain_selection_summary.csv", index=False)

        with open(fold_dir / "selected_domain_metadata.json", "w") as file:
            json.dump(
                {
                    "domain_names": selected_domain_names,
                    "domain_indices": {
                        domain: indices.tolist()
                        for domain, indices in selected_domain_indices.items()
                    },
                },
                file,
                indent=2,
            )

        # ------------------------------------------------------------------
        # UNIFORM DARN
        # ------------------------------------------------------------------
        darn_validation_seed_predictions: List[np.ndarray] = []
        darn_test_seed_predictions: List[np.ndarray] = []
        darn_models: List[UniformDARNClassifier] = []

        for seed_offset in range(config.darn_seeds_per_fold):
            seed = fold_seed + seed_offset
            set_seed(seed)
            print("\n" + "-" * 100)
            print(f"Uniform DARN | Fold {fold_number} | Seed {seed}")
            print("-" * 100)

            darn_model = UniformDARNClassifier(
                domain_slices=selected_domain_slices,
                hidden_dim=config.darn_hidden_dim,
                dropout=config.darn_dropout,
                feature_dropout=config.darn_feature_dropout,
            )

            darn_model, darn_history, darn_best = train_torch_binary_model(
                darn_model,
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
                learning_rate=config.darn_learning_rate,
                weight_decay=config.darn_weight_decay,
                batch_size=config.darn_batch_size,
                max_epochs=config.darn_max_epochs,
                patience=config.darn_patience,
                pos_weight=config.darn_pos_weight,
                focal_gamma=config.darn_focal_gamma,
                seed=seed,
                label=f"DARN F{fold_number} S{seed}",
            )

            device = next(darn_model.parameters()).device
            validation_prob = predict_torch_model(
                darn_model,
                X_fold_validation,
                config.prediction_batch_size,
                device,
            )
            test_prob = predict_torch_model(
                darn_model,
                X_fold_test,
                config.prediction_batch_size,
                device,
            )

            darn_validation_seed_predictions.append(validation_prob)
            darn_test_seed_predictions.append(test_prob)
            darn_models.append(darn_model)

            torch.save(
                darn_model.state_dict(),
                fold_dir / f"uniform_darn_seed_{seed}.pt",
            )
            darn_history.to_csv(
                fold_dir / f"uniform_darn_history_seed_{seed}.csv",
                index=False,
            )

            fold_rows.append(
                {
                    "fold": fold_number,
                    "model": "Uniform DARN",
                    "seed": seed,
                    "validation_AUROC": roc_auc_score(
                        y_fold_validation, validation_prob
                    ),
                    "validation_AUPRC": average_precision_score(
                        y_fold_validation, validation_prob
                    ),
                    **darn_best,
                }
            )

        darn_validation_prob = np.mean(
            np.vstack(darn_validation_seed_predictions), axis=0
        )
        darn_test_prob = np.mean(
            np.vstack(darn_test_seed_predictions), axis=0
        )

        oof_probabilities["Uniform DARN"][fold_validation_indices] = (
            darn_validation_prob
        )
        test_fold_probabilities["Uniform DARN"].append(darn_test_prob)
        (
            darn_validation_calibrated,
            darn_test_calibrated,
            darn_fold_calibrator,
        ) = cross_fitted_fold_calibration(
            y_fold_validation,
            darn_validation_prob,
            darn_test_prob,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=fold_seed + 11,
        )
        calibrated_oof_probabilities["Uniform DARN"][fold_validation_indices] = (
            darn_validation_calibrated
        )
        test_fold_calibrated_probabilities["Uniform DARN"].append(
            darn_test_calibrated
        )
        joblib.dump(
            darn_fold_calibrator,
            fold_dir / "calibrator_uniform_darn.joblib",
        )

        # Complementary input-domain masking sensitivity analysis.
        # This is not retraining: the already-fitted fold model is retained and
        # one selected transformed domain is replaced by the reference value 0.
        if config.run_input_domain_masking:
            for domain in selected_domain_names:
                masked_test = X_fold_test.copy()
                masked_test[:, selected_domain_indices[domain]] = 0.0

                seed_predictions = []
                for darn_model in darn_models:
                    device = next(darn_model.parameters()).device
                    seed_predictions.append(
                        predict_torch_model(
                            darn_model,
                            masked_test,
                            config.prediction_batch_size,
                            device,
                        )
                    )

                fold_masked_probability = np.mean(
                    np.vstack(seed_predictions), axis=0
                )
                darn_ablated_test_probabilities.setdefault(domain, []).append(
                    fold_masked_probability
                )

        if config.run_gradient_shap and darn_models:
            fold_shap = aggregate_gradient_shap_for_fold(
                darn_models[0],
                X_fold_train,
                X_fold_test,
                y_test,
                selected_feature_names,
                max_samples=config.max_shap_n,
                n_samples=config.shap_gradient_samples,
                seed=fold_seed,
            )
            if fold_shap is not None:
                fold_shap["fold"] = fold_number
                gradient_shap_tables.append(fold_shap)

        # Free the first set of GPU models after ablation/SHAP.
        for darn_model in darn_models:
            darn_model.to("cpu")
        del darn_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # ASA-ONLY CLINICAL BASELINE
        # ------------------------------------------------------------------
        print("\nTraining ASA-only logistic baseline...")
        if "asa" not in X_fold_train_raw.columns:
            raise ValueError("ASA-only baseline requested, but 'asa' is unavailable.")
        asa_model, asa_validation_prob, asa_test_prob = fit_asa_only_baseline(
            X_fold_train_raw["asa"],
            y_fold_train,
            X_fold_validation_raw["asa"],
            X_test["asa"],
            config,
            seed=fold_seed,
        )
        oof_probabilities["ASA-only Logistic"][fold_validation_indices] = asa_validation_prob
        test_fold_probabilities["ASA-only Logistic"].append(asa_test_prob)
        (
            asa_validation_calibrated,
            asa_test_calibrated,
            asa_fold_calibrator,
        ) = cross_fitted_fold_calibration(
            y_fold_validation,
            asa_validation_prob,
            asa_test_prob,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=fold_seed + 12,
        )
        calibrated_oof_probabilities["ASA-only Logistic"][fold_validation_indices] = (
            asa_validation_calibrated
        )
        test_fold_calibrated_probabilities["ASA-only Logistic"].append(
            asa_test_calibrated
        )
        joblib.dump(asa_model, fold_dir / "asa_only_logistic.joblib")
        joblib.dump(asa_fold_calibrator, fold_dir / "calibrator_asa_only_logistic.joblib")
        fold_rows.append(
            {
                "fold": fold_number,
                "model": "ASA-only Logistic",
                "seed": fold_seed,
                "validation_AUROC": roc_auc_score(y_fold_validation, asa_validation_prob),
                "validation_AUPRC": average_precision_score(y_fold_validation, asa_validation_prob),
            }
        )

        # ------------------------------------------------------------------
        # LOGISTIC REGRESSION
        # ------------------------------------------------------------------
        print("\nTraining logistic regression baseline...")
        logistic_model = fit_logistic_baseline(
            X_fold_train,
            y_fold_train,
            config,
            seed=fold_seed,
        )
        logistic_validation_prob = logistic_model.predict_proba(
            X_fold_validation
        )[:, 1]
        logistic_test_prob = logistic_model.predict_proba(X_fold_test)[:, 1]

        oof_probabilities["Logistic Regression"][fold_validation_indices] = (
            logistic_validation_prob
        )
        test_fold_probabilities["Logistic Regression"].append(
            logistic_test_prob
        )
        (
            logistic_validation_calibrated,
            logistic_test_calibrated,
            logistic_fold_calibrator,
        ) = cross_fitted_fold_calibration(
            y_fold_validation,
            logistic_validation_prob,
            logistic_test_prob,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=fold_seed + 13,
        )
        calibrated_oof_probabilities["Logistic Regression"][fold_validation_indices] = (
            logistic_validation_calibrated
        )
        test_fold_calibrated_probabilities["Logistic Regression"].append(
            logistic_test_calibrated
        )
        joblib.dump(logistic_model, fold_dir / "logistic_regression.joblib")
        joblib.dump(
            logistic_fold_calibrator,
            fold_dir / "calibrator_logistic_regression.joblib",
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "model": "Logistic Regression",
                "seed": fold_seed,
                "validation_AUROC": roc_auc_score(
                    y_fold_validation, logistic_validation_prob
                ),
                "validation_AUPRC": average_precision_score(
                    y_fold_validation, logistic_validation_prob
                ),
            }
        )

        # ------------------------------------------------------------------
        # XGBOOST
        # ------------------------------------------------------------------
        if "XGBoost" in model_names:
            print("\nTraining XGBoost baseline...")
            xgb_model = make_xgboost_model(
                y_fold_train,
                config,
                seed=fold_seed,
            )
            xgb_model = fit_xgboost_with_early_stopping(
                xgb_model,
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
                config.xgb_early_stopping_rounds,
            )

            xgb_validation_prob = xgb_model.predict_proba(
                X_fold_validation
            )[:, 1]
            xgb_test_prob = xgb_model.predict_proba(X_fold_test)[:, 1]

            if config.run_xgb_contributions:
                rng = np.random.default_rng(fold_seed)
                positive_indices = np.where(y_test == 1)[0]
                negative_indices = np.where(y_test == 0)[0]
                remaining = max(0, config.max_shap_n - len(positive_indices))
                sampled_negative = rng.choice(
                    negative_indices,
                    size=min(remaining, len(negative_indices)),
                    replace=False,
                )
                contribution_indices = np.concatenate([positive_indices, sampled_negative])
                dmatrix = xgb.DMatrix(X_fold_test[contribution_indices], feature_names=[str(name) for name in selected_feature_names])
                contributions = xgb_model.get_booster().predict(dmatrix, pred_contribs=True)
                mean_absolute = np.abs(contributions[:, :-1]).mean(axis=0)
                xgb_contribution_tables.append(
                    pd.DataFrame(
                        {
                            "fold": fold_number,
                            "feature": selected_feature_names.astype(str),
                            "mean_absolute_tree_contribution": mean_absolute,
                        }
                    )
                )

            oof_probabilities["XGBoost"][fold_validation_indices] = (
                xgb_validation_prob
            )
            test_fold_probabilities["XGBoost"].append(xgb_test_prob)
            (
                xgb_validation_calibrated,
                xgb_test_calibrated,
                xgb_fold_calibrator,
            ) = cross_fitted_fold_calibration(
                y_fold_validation,
                xgb_validation_prob,
                xgb_test_prob,
                n_splits=config.platt_crossfit_folds,
                c_value=config.platt_c,
                random_state=fold_seed + 14,
            )
            calibrated_oof_probabilities["XGBoost"][fold_validation_indices] = (
                xgb_validation_calibrated
            )
            test_fold_calibrated_probabilities["XGBoost"].append(
                xgb_test_calibrated
            )
            joblib.dump(xgb_model, fold_dir / "xgboost.joblib")
            joblib.dump(xgb_fold_calibrator, fold_dir / "calibrator_xgboost.joblib")

            fold_rows.append(
                {
                    "fold": fold_number,
                    "model": "XGBoost",
                    "seed": fold_seed,
                    "validation_AUROC": roc_auc_score(
                        y_fold_validation, xgb_validation_prob
                    ),
                    "validation_AUPRC": average_precision_score(
                        y_fold_validation, xgb_validation_prob
                    ),
                    "best_iteration": float(
                        getattr(
                            xgb_model,
                            "strict_oof_best_iteration_",
                            np.nan,
                        )
                    ),
                    "selected_n_estimators": float(
                        getattr(
                            xgb_model,
                            "strict_oof_n_estimators_",
                            np.nan,
                        )
                    ),
                }
            )

        # ------------------------------------------------------------------
        # STANDARD MLP
        # ------------------------------------------------------------------
        print("\nTraining MLP baseline...")
        set_seed(fold_seed + 50)
        mlp_model = MLPBaseline(
            input_dim=X_fold_train.shape[1],
            hidden_dim=config.mlp_hidden_dim,
            dropout=config.mlp_dropout,
        )

        mlp_model, mlp_history, mlp_best = train_torch_binary_model(
            mlp_model,
            X_fold_train,
            y_fold_train,
            X_fold_validation,
            y_fold_validation,
            learning_rate=config.mlp_learning_rate,
            weight_decay=config.mlp_weight_decay,
            batch_size=config.mlp_batch_size,
            max_epochs=config.mlp_max_epochs,
            patience=config.mlp_patience,
            pos_weight=config.mlp_pos_weight,
            focal_gamma=0.0,
            seed=fold_seed + 50,
            label=f"MLP F{fold_number}",
        )

        device = next(mlp_model.parameters()).device
        mlp_validation_prob = predict_torch_model(
            mlp_model,
            X_fold_validation,
            config.prediction_batch_size,
            device,
        )
        mlp_test_prob = predict_torch_model(
            mlp_model,
            X_fold_test,
            config.prediction_batch_size,
            device,
        )

        oof_probabilities["MLP"][fold_validation_indices] = mlp_validation_prob
        test_fold_probabilities["MLP"].append(mlp_test_prob)
        (
            mlp_validation_calibrated,
            mlp_test_calibrated,
            mlp_fold_calibrator,
        ) = cross_fitted_fold_calibration(
            y_fold_validation,
            mlp_validation_prob,
            mlp_test_prob,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=fold_seed + 15,
        )
        calibrated_oof_probabilities["MLP"][fold_validation_indices] = (
            mlp_validation_calibrated
        )
        test_fold_calibrated_probabilities["MLP"].append(mlp_test_calibrated)

        torch.save(mlp_model.state_dict(), fold_dir / "mlp_baseline.pt")
        joblib.dump(mlp_fold_calibrator, fold_dir / "calibrator_mlp.joblib")
        mlp_history.to_csv(fold_dir / "mlp_history.csv", index=False)

        fold_rows.append(
            {
                "fold": fold_number,
                "model": "MLP",
                "seed": fold_seed + 50,
                "validation_AUROC": roc_auc_score(
                    y_fold_validation, mlp_validation_prob
                ),
                "validation_AUPRC": average_precision_score(
                    y_fold_validation, mlp_validation_prob
                ),
                **mlp_best,
            }
        )

        mlp_model.to("cpu")
        del mlp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"Fold {fold_number} completed in "
            f"{(time.time() - fold_start) / 60.0:.2f} minutes."
        )

    fold_performance = pd.DataFrame(fold_rows)
    fold_performance.to_csv(run_dir / "fold_validation_performance.csv", index=False)

    # Ensure all OOF rows were predicted exactly once.
    for model_name, predictions in oof_probabilities.items():
        if not np.isfinite(predictions).all():
            missing = int((~np.isfinite(predictions)).sum())
            raise RuntimeError(f"{model_name} has {missing} missing OOF predictions.")

    raw_test_probabilities = {
        model_name: np.mean(np.vstack(fold_predictions), axis=0)
        for model_name, fold_predictions in test_fold_probabilities.items()
    }
    calibrated_test_probabilities: Dict[str, np.ndarray] = {
        model_name: np.mean(np.vstack(fold_predictions), axis=0)
        for model_name, fold_predictions in test_fold_calibrated_probabilities.items()
    }

    thresholds: Dict[str, float] = {}
    selective_thresholds: Dict[str, float] = {}
    calibrators: Dict[str, Any] = {}

    print("\n" + "=" * 100)
    print("CROSS-FITTED FOLD CALIBRATION AND DEVELOPMENT THRESHOLDS")
    print("=" * 100)

    for model_name in base_model_names:
        calibrated_oof = calibrated_oof_probabilities[model_name]
        calibrated_test = calibrated_test_probabilities[model_name]

        if not np.isfinite(calibrated_oof).all():
            missing = int((~np.isfinite(calibrated_oof)).sum())
            raise RuntimeError(
                f"{model_name} has {missing} missing calibrated OOF predictions."
            )
        if not np.isfinite(calibrated_test).all():
            missing = int((~np.isfinite(calibrated_test)).sum())
            raise RuntimeError(
                f"{model_name} has {missing} missing calibrated test predictions."
            )

        threshold = select_operating_threshold(
            y_development,
            calibrated_oof,
            strategy=config.threshold_strategy,
            target_sensitivity=config.target_sensitivity,
            target_specificity=config.selective_target_specificity,
        )
        selective_threshold = select_operating_threshold(
            y_development,
            calibrated_oof,
            strategy="specificity",
            target_sensitivity=config.target_sensitivity,
            target_specificity=config.selective_target_specificity,
        )

        thresholds[model_name] = threshold
        selective_thresholds[model_name] = selective_threshold

        print(
            f"{model_name:<30} | "
            f"OOF AUROC={roc_auc_score(y_development, calibrated_oof):.4f} | "
            f"OOF AUPRC={average_precision_score(y_development, calibrated_oof):.4f} | "
            f"Sensitivity threshold={threshold:.8f} | "
            f"Selective threshold={selective_threshold:.8f}"
        )

    # ------------------------------------------------------------------
    # LEAKAGE-SAFE DARN-XGBOOST HYBRIDS
    # ------------------------------------------------------------------
    hybrid_metadata: Dict[str, Any] = {}

    if config.run_hybrid and "XGBoost" in base_model_names:
        darn_oof_cal = calibrated_oof_probabilities["Uniform DARN"]
        xgb_oof_cal = calibrated_oof_probabilities["XGBoost"]
        darn_test_cal = calibrated_test_probabilities["Uniform DARN"]
        xgb_test_cal = calibrated_test_probabilities["XGBoost"]

        # 1) OOF-optimized convex probability blend.
        best_darn_weight, blend_search = optimize_darn_xgb_blend(
            y_development,
            darn_oof_cal,
            xgb_oof_cal,
            grid_size=config.hybrid_blend_grid_size,
            min_darn_weight=config.hybrid_min_darn_weight,
            max_darn_weight=config.hybrid_max_darn_weight,
        )
        blend_name = "Hybrid DARN-XGB Blend"
        blend_oof_raw = (
            best_darn_weight * darn_oof_cal
            + (1.0 - best_darn_weight) * xgb_oof_cal
        )
        blend_test_raw = (
            best_darn_weight * darn_test_cal
            + (1.0 - best_darn_weight) * xgb_test_cal
        )
        (
            blend_oof,
            blend_test,
            blend_calibrator,
        ) = cross_fitted_fold_calibration(
            y_development,
            blend_oof_raw,
            blend_test_raw,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=config.random_state + 801,
        )
        blend_threshold = select_operating_threshold(
            y_development,
            blend_oof,
            strategy=config.threshold_strategy,
            target_sensitivity=config.target_sensitivity,
            target_specificity=config.selective_target_specificity,
        )
        blend_selective_threshold = select_operating_threshold(
            y_development,
            blend_oof,
            strategy="specificity",
            target_specificity=config.selective_target_specificity,
        )

        model_names.append(blend_name)
        oof_probabilities[blend_name] = blend_oof_raw
        raw_test_probabilities[blend_name] = blend_test_raw
        calibrated_oof_probabilities[blend_name] = blend_oof
        calibrated_test_probabilities[blend_name] = blend_test
        thresholds[blend_name] = blend_threshold
        selective_thresholds[blend_name] = blend_selective_threshold
        calibrators[blend_name] = blend_calibrator
        blend_search.to_csv(run_dir / "hybrid_blend_weight_search.csv", index=False)
        joblib.dump(blend_calibrator, run_dir / "calibrator_hybrid_darn_xgb_blend.joblib")

        hybrid_metadata[blend_name] = {
            "darn_weight": float(best_darn_weight),
            "xgboost_weight": float(1.0 - best_darn_weight),
            "selection_source": "development OOF predictions only",
        }

        print(
            f"{blend_name:<30} | "
            f"OOF AUROC={roc_auc_score(y_development, blend_oof):.4f} | "
            f"OOF AUPRC={average_precision_score(y_development, blend_oof):.4f} | "
            f"Threshold={blend_threshold:.8f} | "
            f"DARN weight={best_darn_weight:.3f}"
        )

        # 2) Cross-fitted logistic stack using only DARN and XGBoost outputs.
        stack_name = "Hybrid DARN-XGB Stack"
        stack_oof_raw, stack_test_raw, final_stacker = (
            cross_fitted_darn_xgb_stacker(
                y_development,
                darn_oof_cal,
                xgb_oof_cal,
                darn_test_cal,
                xgb_test_cal,
                n_folds=config.hybrid_stack_folds,
                c_value=config.hybrid_stack_c,
                class_weight_balanced=config.hybrid_stack_class_weight_balanced,
                random_state=config.random_state + 707,
            )
        )
        (
            stack_oof,
            stack_test,
            stack_calibrator,
        ) = cross_fitted_fold_calibration(
            y_development,
            stack_oof_raw,
            stack_test_raw,
            n_splits=config.platt_crossfit_folds,
            c_value=config.platt_c,
            random_state=config.random_state + 802,
        )
        stack_threshold = select_operating_threshold(
            y_development,
            stack_oof,
            strategy=config.threshold_strategy,
            target_sensitivity=config.target_sensitivity,
            target_specificity=config.selective_target_specificity,
        )
        stack_selective_threshold = select_operating_threshold(
            y_development,
            stack_oof,
            strategy="specificity",
            target_specificity=config.selective_target_specificity,
        )

        model_names.append(stack_name)
        oof_probabilities[stack_name] = stack_oof_raw
        raw_test_probabilities[stack_name] = stack_test_raw
        calibrated_oof_probabilities[stack_name] = stack_oof
        calibrated_test_probabilities[stack_name] = stack_test
        thresholds[stack_name] = stack_threshold
        selective_thresholds[stack_name] = stack_selective_threshold
        calibrators[stack_name] = stack_calibrator
        joblib.dump(final_stacker, run_dir / "hybrid_darn_xgb_final_stacker.joblib")
        pd.DataFrame({
            "meta_feature": ["DARN logit", "XGBoost logit", "logit difference", "mean logit", "absolute logit difference"],
            "coefficient": final_stacker.coef_.ravel(),
        }).to_csv(run_dir / "hybrid_stack_coefficients.csv", index=False)
        joblib.dump(stack_calibrator, run_dir / "calibrator_hybrid_darn_xgb_stack.joblib")

        hybrid_metadata[stack_name] = {
            "stack_features": [
                "DARN logit",
                "XGBoost logit",
                "logit difference",
                "mean logit",
                "absolute logit difference",
            ],
            "meta_folds": int(config.hybrid_stack_folds),
            "stack_C": float(config.hybrid_stack_c),
            "selection_source": "cross-fitted development OOF predictions only",
        }

        print(
            f"{stack_name:<30} | "
            f"OOF AUROC={roc_auc_score(y_development, stack_oof):.4f} | "
            f"OOF AUPRC={average_precision_score(y_development, stack_oof):.4f} | "
            f"Threshold={stack_threshold:.8f}"
        )

        with open(run_dir / "hybrid_metadata.json", "w") as file:
            json.dump(hybrid_metadata, file, indent=2, default=json_default)

    # ------------------------------------------------------------------
    # TRANSPORT DEVELOPMENT OPERATING POINTS TO ENSEMBLED TEST SCORES
    # ------------------------------------------------------------------
    development_thresholds = dict(thresholds)
    development_selective_thresholds = dict(selective_thresholds)
    applied_thresholds: Dict[str, float] = {}
    applied_selective_thresholds: Dict[str, float] = {}
    threshold_transport_rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        applied_threshold, metadata = transport_threshold_to_test(
            calibrated_oof_probabilities[model_name],
            development_thresholds[model_name],
            calibrated_test_probabilities[model_name],
            mode=config.threshold_transport_mode,
            max_rate_ratio=config.threshold_transport_rate_ratio,
            max_absolute_gap=config.threshold_transport_absolute_gap,
            degenerate_rate=config.threshold_transport_degenerate_rate,
        )
        applied_selective_threshold, selective_metadata = transport_threshold_to_test(
            calibrated_oof_probabilities[model_name],
            development_selective_thresholds[model_name],
            calibrated_test_probabilities[model_name],
            mode=config.threshold_transport_mode,
            max_rate_ratio=config.threshold_transport_rate_ratio,
            max_absolute_gap=config.threshold_transport_absolute_gap,
            degenerate_rate=config.threshold_transport_degenerate_rate,
        )
        applied_thresholds[model_name] = applied_threshold
        applied_selective_thresholds[model_name] = applied_selective_threshold

        threshold_transport_rows.append(
            {
                "model": model_name,
                "operating_point": "High sensitivity",
                **metadata,
            }
        )
        threshold_transport_rows.append(
            {
                "model": model_name,
                "operating_point": "Selective specificity",
                **selective_metadata,
            }
        )

    threshold_transport_table = pd.DataFrame(threshold_transport_rows)
    threshold_transport_table.to_csv(
        run_dir / "threshold_transport_audit.csv", index=False
    )

    print("\n" + "=" * 100)
    print("THRESHOLD TRANSPORT AUDIT")
    print("=" * 100)
    print(
        threshold_transport_table[
            [
                "model",
                "operating_point",
                "development_alert_rate",
                "test_alert_rate_at_absolute_threshold",
                "applied_test_alert_rate",
                "transport_method",
            ]
        ].to_string(index=False)
    )

    # All downstream test operating-point analyses use the applied thresholds.
    # The original development thresholds remain available in
    # threshold_transport_audit.csv.
    thresholds = applied_thresholds
    selective_thresholds = applied_selective_thresholds

    # Test metrics
    metric_rows: List[Dict[str, Any]] = []
    selective_metric_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 100)
    print("UNTOUCHED TEST PERFORMANCE")
    print("=" * 100)

    for model_name in model_names:
        metrics = calculate_binary_metrics(
            y_test,
            calibrated_test_probabilities[model_name],
            thresholds[model_name],
        )
        metrics["model"] = model_name
        metrics["Operating_point"] = f"Development OOF sensitivity >= {config.target_sensitivity:.2f}"
        high_transport = threshold_transport_table.loc[
            (threshold_transport_table["model"] == model_name)
            & (threshold_transport_table["operating_point"] == "High sensitivity")
        ].iloc[0]
        metrics["Development_threshold"] = float(development_thresholds[model_name])
        metrics["Applied_test_threshold"] = float(thresholds[model_name])
        metrics["Development_alert_rate"] = float(high_transport["development_alert_rate"])
        metrics["Applied_test_alert_rate"] = float(high_transport["applied_test_alert_rate"])
        metrics["Threshold_transport_method"] = str(high_transport["transport_method"])
        metric_rows.append(metrics)

        selective_metrics = calculate_binary_metrics(
            y_test,
            calibrated_test_probabilities[model_name],
            selective_thresholds[model_name],
        )
        selective_metrics["model"] = model_name
        selective_metrics["Operating_point"] = f"Development OOF specificity >= {config.selective_target_specificity:.2f}"
        selective_transport = threshold_transport_table.loc[
            (threshold_transport_table["model"] == model_name)
            & (threshold_transport_table["operating_point"] == "Selective specificity")
        ].iloc[0]
        selective_metrics["Development_threshold"] = float(
            development_selective_thresholds[model_name]
        )
        selective_metrics["Applied_test_threshold"] = float(
            selective_thresholds[model_name]
        )
        selective_metrics["Development_alert_rate"] = float(
            selective_transport["development_alert_rate"]
        )
        selective_metrics["Applied_test_alert_rate"] = float(
            selective_transport["applied_test_alert_rate"]
        )
        selective_metrics["Threshold_transport_method"] = str(
            selective_transport["transport_method"]
        )
        selective_metric_rows.append(selective_metrics)

        print(
            f"{model_name:<30} | "
            f"AUROC={metrics['AUROC']:.4f} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"Brier={metrics['Brier']:.6f} | "
            f"Accuracy={metrics['Accuracy']:.3f} | "
            f"Recall={metrics['Recall']:.3f} | "
            f"Specificity={metrics['Specificity']:.3f} | "
            f"PPV={metrics['Precision']:.4f}"
        )

    metrics_table = pd.DataFrame(metric_rows)
    preferred_columns = [
        "model",
        "AUROC",
        "AUPRC",
        "AUPRC_fold_over_baseline",
        "Brier",
        "Calibration_slope",
        "Calibration_intercept",
        "ECE",
        "Hosmer_Lemeshow_chi2",
        "Hosmer_Lemeshow_df",
        "Hosmer_Lemeshow_p",
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "False_positive_rate",
        "False_negative_rate",
        "F1",
        "Threshold",
        "Development_threshold",
        "Applied_test_threshold",
        "Development_alert_rate",
        "Applied_test_alert_rate",
        "Threshold_transport_method",
        "TP",
        "FN",
        "FP",
        "TN",
        "Top_1pct_events",
        "Top_2pct_events",
        "Top_5pct_events",
        "Top_10pct_events",
    ]
    remaining_columns = [
        column for column in metrics_table.columns if column not in preferred_columns
    ]
    metrics_table = metrics_table[
        [column for column in preferred_columns if column in metrics_table]
        + remaining_columns
    ].sort_values("AUPRC", ascending=False)
    metrics_table.to_csv(run_dir / "test_model_metrics.csv", index=False)
    selective_metrics_table = pd.DataFrame(selective_metric_rows).sort_values("AUPRC", ascending=False)
    selective_metrics_table.to_csv(run_dir / "test_model_metrics_selective_operating_point.csv", index=False)
    pd.concat([metrics_table, selective_metrics_table], ignore_index=True).to_csv(
        run_dir / "test_model_metrics_all_operating_points.csv", index=False
    )

    # Prediction files
    test_prediction_table = pd.DataFrame(
        {
            "final_cohort_row_position": test_positions,
            "source_row_position": source_row_positions[test_positions],
            "y_true": y_test,
        }
    )
    oof_prediction_table = pd.DataFrame(
        {
            "development_row_position": np.arange(len(y_development)),
            "final_cohort_row_position": development_positions,
            "source_row_position": source_row_positions[development_positions],
            "y_true": y_development,
        }
    )

    for model_name in model_names:
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
        test_prediction_table[f"{safe_name}_raw_probability"] = raw_test_probabilities[
            model_name
        ]
        test_prediction_table[f"{safe_name}_calibrated_probability"] = (
            calibrated_test_probabilities[model_name]
        )
        test_prediction_table[f"{safe_name}_prediction_high_sensitivity"] = (
            calibrated_test_probabilities[model_name] >= thresholds[model_name]
        ).astype(int)
        test_prediction_table[f"{safe_name}_prediction_selective"] = (
            calibrated_test_probabilities[model_name] >= selective_thresholds[model_name]
        ).astype(int)

        oof_prediction_table[f"{safe_name}_raw_probability"] = oof_probabilities[
            model_name
        ]
        oof_prediction_table[f"{safe_name}_calibrated_probability"] = (
            calibrated_oof_probabilities[model_name]
        )

    test_prediction_table.to_csv(run_dir / "test_predictions_all_models.csv", index=False)
    oof_prediction_table.to_csv(run_dir / "development_oof_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # COMPLEMENTARY INPUT-DOMAIN MASKING SENSITIVITY
    # ------------------------------------------------------------------
    darn_baseline_raw = raw_test_probabilities["Uniform DARN"]
    darn_baseline_auc = roc_auc_score(y_test, darn_baseline_raw)
    darn_baseline_ap = average_precision_score(y_test, darn_baseline_raw)
    masking_rows: List[Dict[str, Any]] = []

    for domain, fold_probabilities in darn_ablated_test_probabilities.items():
        masked_raw = np.mean(np.vstack(fold_probabilities), axis=0)
        masked_auc = roc_auc_score(y_test, masked_raw)
        masked_ap = average_precision_score(y_test, masked_raw)
        masking_rows.append(
            {
                "analysis_method": "Fitted-model input-domain masking",
                "domain": domain,
                "baseline_AUROC": darn_baseline_auc,
                "masked_AUROC": masked_auc,
                # Backward-compatible aliases used by the current Figure 5
                # plotting script. The analysis_method field makes clear that
                # these are masking results, not retrained ablations.
                "ablated_AUROC": masked_auc,
                "AUROC_drop": darn_baseline_auc - masked_auc,
                "baseline_AUPRC": darn_baseline_ap,
                "masked_AUPRC": masked_ap,
                "ablated_AUPRC": masked_ap,
                "AUPRC_drop": darn_baseline_ap - masked_ap,
            }
        )

    input_domain_masking = pd.DataFrame(masking_rows)
    if not input_domain_masking.empty:
        input_domain_masking = input_domain_masking.sort_values(
            "AUPRC_drop",
            ascending=False,
        )

    input_domain_masking.to_csv(
        run_dir / "darn_input_domain_masking.csv",
        index=False,
    )

    # Backward-compatible filename for older plotting scripts. The method
    # column prevents this file from being mistaken for retraining.
    input_domain_masking.to_csv(
        run_dir / "darn_domain_ablation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TRUE LEAVE-ONE-DOMAIN-OUT RETRAINING
    # ------------------------------------------------------------------
    if config.run_leave_one_domain_out_retraining:
        leave_one_domain_out = run_leave_one_domain_out_retraining(
            X_development=X_development,
            y_development=y_development,
            X_test=X_test,
            y_test=y_test,
            fold_splits=fold_splits,
            full_test_probabilities=calibrated_test_probabilities,
            config=config,
            run_dir=run_dir,
        )
    else:
        leave_one_domain_out = {
            "domains": [],
            "variant_results": [],
            "metrics": pd.DataFrame(),
            "paired_bootstrap": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "figure": None,
            "output_directory": None,
        }

    # Optional GradientSHAP aggregation by transformed feature name.
    if gradient_shap_tables:
        gradient_shap_all = pd.concat(gradient_shap_tables, ignore_index=True)
        gradient_shap_summary = (
            gradient_shap_all.groupby("feature", as_index=False)[
                "mean_absolute_gradientshap"
            ]
            .mean()
            .sort_values("mean_absolute_gradientshap", ascending=False)
        )
        gradient_shap_all.to_csv(
            run_dir / "darn_gradientshap_by_fold.csv", index=False
        )
        gradient_shap_summary.to_csv(
            run_dir / "darn_gradientshap_summary.csv", index=False
        )

    if xgb_contribution_tables:
        xgb_contributions = pd.concat(xgb_contribution_tables, ignore_index=True)
        xgb_contributions.to_csv(run_dir / "xgboost_tree_contributions_by_fold.csv", index=False)
        (
            xgb_contributions.groupby("feature", as_index=False)["mean_absolute_tree_contribution"]
            .mean()
            .sort_values("mean_absolute_tree_contribution", ascending=False)
            .to_csv(run_dir / "xgboost_tree_contributions_summary.csv", index=False)
        )

    if variance_filter_tables:
        variance_all = pd.concat(variance_filter_tables, ignore_index=True)
        variance_all.to_csv(run_dir / "variance_filter_all_folds.csv", index=False)
        variance_all.loc[~variance_all["retained"]].to_csv(
            run_dir / "variance_filtered_out_features_all_folds.csv", index=False
        )

    # Clinical utility, subgroup fairness, and error analysis requested by reviewers.
    risk_concentration = build_risk_concentration_table(y_test, calibrated_test_probabilities)
    risk_concentration.to_csv(run_dir / "risk_concentration.csv", index=False)

    decision_curve = decision_curve_analysis(
        y_test,
        calibrated_test_probabilities,
        min_threshold=config.decision_curve_min_threshold,
        max_threshold=config.decision_curve_max_threshold,
        n_points=config.decision_curve_points,
    )
    decision_curve.to_csv(run_dir / "decision_curve_analysis.csv", index=False)
    plt.figure(figsize=(9, 7))
    for model_name, group in decision_curve.groupby("model"):
        plt.plot(group["threshold_probability"], group["net_benefit"], linewidth=2, label=model_name)
    plt.xscale("log")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "Decision_curve_analysis.png", dpi=400)
    plt.close()

    subgroup_frame = derive_subgroup_variables(X_test.reset_index(drop=True))
    subgroup_performance, subgroup_gaps = subgroup_performance_with_bootstrap(
        y_test,
        calibrated_test_probabilities,
        {
            "High sensitivity": thresholds,
            "Selective specificity": selective_thresholds,
        },
        subgroup_frame,
        n_bootstrap=config.subgroup_bootstrap,
        random_state=config.random_state + 1701,
        min_events=config.subgroup_min_events,
        min_nonevents=config.subgroup_min_nonevents,
    )
    subgroup_performance.to_csv(run_dir / "subgroup_performance_with_bootstrap_CI.csv", index=False)
    subgroup_gaps.to_csv(run_dir / "subgroup_fairness_gaps_equalized_odds.csv", index=False)

    primary_model = (
        config.primary_model_name
        if config.primary_model_name in model_names
        else (
            "Hybrid DARN-XGB Blend"
            if "Hybrid DARN-XGB Blend" in model_names
            else metrics_table.iloc[0]["model"]
        )
    )

    error_cases, error_numeric, error_categorical = create_error_analysis(
        X_test.reset_index(drop=True),
        y_test,
        calibrated_test_probabilities,
        primary_model=primary_model,
        primary_threshold=thresholds[primary_model],
        selective_threshold=selective_thresholds[primary_model],
    )
    error_cases.to_csv(
        run_dir / "error_analysis_cases_primary_model.csv",
        index=False,
    )
    error_numeric.to_csv(
        run_dir / "error_analysis_numeric_summary.csv",
        index=False,
    )
    error_categorical.to_csv(
        run_dir / "error_analysis_categorical_summary.csv",
        index=False,
    )

    if config.run_false_negative_review:
        false_negative_review = create_false_negative_case_review(
            X_test.reset_index(drop=True),
            y_test,
            calibrated_test_probabilities[primary_model],
            thresholds[primary_model],
        )
        false_negative_review.to_csv(
            run_dir / "false_negative_case_review_deidentified.csv",
            index=False,
        )
    else:
        false_negative_review = pd.DataFrame()

    if config.run_tail_risk_strata_calibration:
        tail_calibration = build_prespecified_tail_calibration_table(
            y_test,
            calibrated_test_probabilities[primary_model],
        )
        tail_calibration.to_csv(
            run_dir / "tail_risk_strata_calibration.csv",
            index=False,
        )
        save_prespecified_tail_calibration_figure(
            tail_calibration,
            run_dir
            / "figures"
            / "Tail_risk_strata_calibration.png",
        )
    else:
        tail_calibration = pd.DataFrame()

    eye_ear_column = identify_eye_ear_diagnosis_column(
        X_raw
    )
    eye_ear_adjusted_results = pd.DataFrame()
    eye_ear_retraining = {
        "metrics": pd.DataFrame(),
        "paired_bootstrap": pd.DataFrame(),
        "figure": None,
    }

    if (
        config.run_eye_ear_adjusted_analysis
        and eye_ear_column is not None
    ):
        eye_ear_adjusted_results = (
            eye_ear_adjusted_association_analysis(
                df,
                y,
                eye_ear_column,
                run_dir
                / "eye_ear_adjusted_association.csv",
            )
        )

    if (
        config.run_eye_ear_feature_retraining
        and eye_ear_column is not None
    ):
        eye_ear_output_directory = (
            run_dir
            / "eye_ear_feature_retraining"
        )
        eye_ear_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        eye_ear_variant = fit_retrained_feature_set_variant(
            variant_name="Eye and ear diagnosis feature",
            analysis_type="Targeted feature-removal retraining",
            excluded_raw_columns=[eye_ear_column],
            X_development=X_development,
            y_development=y_development,
            X_test=X_test,
            y_test=y_test,
            fold_splits=fold_splits,
            config=config,
            output_directory=eye_ear_output_directory,
            retrain_hybrid=config.lodo_retrain_hybrid,
        )

        eye_ear_comparisons: List[pd.DataFrame] = []

        for model_name, reduced_probability in (
            eye_ear_variant["probabilities"].items()
        ):
            if model_name not in calibrated_test_probabilities:
                continue

            eye_ear_comparisons.append(
                paired_bootstrap_retraining_difference(
                    y_test,
                    calibrated_test_probabilities[model_name],
                    reduced_probability,
                    analysis="Targeted feature-removal retraining",
                    removed_feature_set=eye_ear_column,
                    model_name=model_name,
                    n_bootstrap=config.lodo_bootstrap,
                    random_state=config.random_state + 7801,
                )
            )

        eye_ear_paired = (
            pd.concat(
                eye_ear_comparisons,
                ignore_index=True,
            )
            if eye_ear_comparisons
            else pd.DataFrame()
        )

        eye_ear_paired.to_csv(
            eye_ear_output_directory
            / "eye_ear_feature_removal_paired_bootstrap.csv",
            index=False,
        )

        eye_ear_retraining = {
            "metrics": eye_ear_variant["metrics"],
            "paired_bootstrap": eye_ear_paired,
            "figure": None,
            "variant": eye_ear_variant,
        }

    with open(run_dir / "baseline_availability.json", "w") as file:
        json.dump(
            {
                "included_baselines": model_names,
                "ASA_only_baseline": "Included",
                "ACS_NSQIP": "Not reconstructed because the complete proprietary/required variable set was unavailable in INSPIRE.",
                "EuroSCORE": "Not applicable to the heterogeneous noncardiac surgical cohort and required variables were unavailable.",
            },
            file,
            indent=2,
        )

    # Bootstrap CIs and paired model differences.
    print("\nComputing stratified bootstrap confidence intervals...")
    bootstrap_table, bootstrap_distributions = bootstrap_model_metrics(
        y_test,
        calibrated_test_probabilities,
        thresholds=thresholds,
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state + 999,
    )
    bootstrap_table.to_csv(run_dir / "bootstrap_confidence_intervals.csv", index=False)

    paired_comparisons = paired_bootstrap_comparisons(
        bootstrap_distributions,
        reference_model="Uniform DARN",
    )
    paired_comparisons.to_csv(
        run_dir / "paired_bootstrap_darn_vs_baselines.csv", index=False
    )

    hybrid_reference_name = (
        primary_model
        if primary_model in calibrated_test_probabilities
        else None
    )

    if hybrid_reference_name is not None:
        paired_hybrid_comparisons = paired_bootstrap_comparisons(
            bootstrap_distributions,
            reference_model=hybrid_reference_name,
        )
        paired_hybrid_comparisons.to_csv(
            run_dir / "paired_bootstrap_hybrid_vs_all.csv", index=False
        )
    else:
        paired_hybrid_comparisons = pd.DataFrame()

    save_comparison_plots(
        y_test,
        calibrated_test_probabilities,
        metrics_table,
        run_dir / "figures",
        calibration_bootstrap=config.calibration_bootstrap,
        calibration_bins=config.calibration_bins,
    )

    # Human-readable summary
    with open(run_dir / "summary.txt", "w") as file:
        file.write("REVIEWER-COMPLETE UNIFORM DARN + HYBRID COMPARISON\n")
        file.write("=" * 80 + "\n")
        file.write(f"Development N: {len(y_development):,}\n")
        file.write(f"Development deaths: {int(y_development.sum()):,}\n")
        file.write(f"Test N: {len(y_test):,}\n")
        file.write(f"Test deaths: {int(y_test.sum()):,}\n\n")
        file.write(metrics_table.to_string(index=False))
        file.write("\n\nSELECTIVE OPERATING POINT\n")
        file.write(selective_metrics_table.to_string(index=False))
        file.write("\n\nRISK CONCENTRATION\n")
        file.write(risk_concentration.to_string(index=False))
        file.write("\n\nDARN INPUT-DOMAIN MASKING SENSITIVITY\n")
        file.write(input_domain_masking.to_string(index=False))
        file.write("\n\nTRUE LEAVE-ONE-DOMAIN-OUT RETRAINING\n")
        file.write(
            leave_one_domain_out["paired_bootstrap"].to_string(index=False)
        )
        file.write("\n\nTARGETED EYE/EAR ADJUSTED ANALYSIS\n")
        file.write(eye_ear_adjusted_results.to_string(index=False))
        file.write("\n\nTARGETED EYE/EAR FEATURE-REMOVAL RETRAINING\n")
        file.write(
            eye_ear_retraining["paired_bootstrap"].to_string(index=False)
        )
        file.write("\n\nPAIRED BOOTSTRAP COMPARISONS: UNIFORM DARN VS OTHERS\n")
        file.write(paired_comparisons.to_string(index=False))
        if not paired_hybrid_comparisons.empty:
            file.write("\n\nPAIRED BOOTSTRAP COMPARISONS: HYBRID VS OTHERS\n")
            file.write(paired_hybrid_comparisons.to_string(index=False))

    elapsed_minutes = (time.time() - overall_start) / 60.0

    print("\n" + "=" * 100)
    print("FINAL RANKING BY TEST AUPRC")
    print("=" * 100)
    print(
        metrics_table[
            [
                "model",
                "AUROC",
                "AUPRC",
                "Brier",
                "Accuracy",
                "Recall",
                "Specificity",
                "Precision",
            ]
        ].to_string(index=False)
    )

    print("\nTop DARN domains by AUPRC change after input masking:")
    print(input_domain_masking.head(10).to_string(index=False))

    if not leave_one_domain_out["paired_bootstrap"].empty:
        print("\nTrue leave-one-domain-out retraining: AUPRC differences")
        print(
            leave_one_domain_out["paired_bootstrap"]
            .loc[
                leave_one_domain_out["paired_bootstrap"]["metric"].eq("AUPRC")
            ]
            .sort_values(
                "difference_favoring_complete_model",
                ascending=False,
            )
            .to_string(index=False)
        )

    print(f"\nCompleted in {elapsed_minutes:.2f} minutes.")
    print("Run directory:", run_dir)

    return {
        "run_dir": str(run_dir),
        "metrics": metrics_table,
        "bootstrap": bootstrap_table,
        "paired_comparisons": paired_comparisons,
        "paired_hybrid_comparisons": paired_hybrid_comparisons,
        "input_domain_masking": input_domain_masking,
        "leave_one_domain_out_retraining": leave_one_domain_out,
        "eye_ear_adjusted_analysis": eye_ear_adjusted_results,
        "eye_ear_feature_retraining": eye_ear_retraining,
        "false_negative_review": false_negative_review,
        "tail_risk_strata_calibration": tail_calibration,
        "selective_metrics": selective_metrics_table,
        "risk_concentration": risk_concentration,
        "decision_curve": decision_curve,
        "subgroup_performance": subgroup_performance,
        "subgroup_fairness_gaps": subgroup_gaps,
        "test_predictions": test_prediction_table,
        "oof_predictions": oof_prediction_table,
    }


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform DARN, leakage-safe DARN-XGBoost hybrids, and fair baselines."
    )

    parser.add_argument("--data_path", default=PipelineConfig.data_path)
    parser.add_argument("--out_root", default=PipelineConfig.out_root)
    parser.add_argument("--folds", type=int, default=PipelineConfig.n_folds)
    parser.add_argument("--bootstrap", type=int, default=PipelineConfig.n_bootstrap)
    parser.add_argument("--darn_seeds_per_fold", type=int, default=PipelineConfig.darn_seeds_per_fold)
    parser.add_argument("--run_gradient_shap", action="store_true", help="Force direct DARN GradientSHAP on.")
    parser.add_argument("--skip_gradient_shap", action="store_true")
    parser.add_argument("--keep_asa6", action="store_true", help="Not recommended; retain ASA 6 encounters.")
    parser.add_argument("--skip_xgboost", action="store_true")
    parser.add_argument("--skip_hybrid", action="store_true")

    parser.add_argument(
        "--primary_model",
        default=PipelineConfig.primary_model_name,
        help="Primary reviewer-facing model for error, subgroup, and calibration analyses.",
    )
    parser.add_argument(
        "--threshold_transport_mode",
        choices=["absolute", "alert_rate", "auto"],
        default=PipelineConfig.threshold_transport_mode,
    )
    parser.add_argument(
        "--skip_leave_one_domain_out",
        action="store_true",
        help="Skip true leave-one-domain-out retraining.",
    )
    parser.add_argument(
        "--lodo_darn_only",
        action="store_true",
        help="Retrain Uniform DARN only; do not rebuild XGBoost and the hybrid blend.",
    )
    parser.add_argument(
        "--lodo_bootstrap",
        type=int,
        default=PipelineConfig.lodo_bootstrap,
    )
    parser.add_argument(
        "--save_lodo_models",
        action="store_true",
        help="Save every leave-one-domain-out fold model and preprocessor.",
    )
    parser.add_argument(
        "--skip_input_domain_masking",
        action="store_true",
    )
    parser.add_argument(
        "--skip_eye_ear_adjusted_analysis",
        action="store_true",
    )
    parser.add_argument(
        "--skip_eye_ear_feature_retraining",
        action="store_true",
    )
    parser.add_argument(
        "--skip_tail_risk_strata_calibration",
        action="store_true",
    )
    parser.add_argument(
        "--skip_false_negative_review",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    config = PipelineConfig(
        data_path=arguments.data_path,
        out_root=arguments.out_root,
        n_folds=arguments.folds,
        n_bootstrap=arguments.bootstrap,
        darn_seeds_per_fold=arguments.darn_seeds_per_fold,
        run_gradient_shap=(not arguments.skip_gradient_shap) or arguments.run_gradient_shap,
        exclude_asa6=not arguments.keep_asa6,
        run_xgboost=not arguments.skip_xgboost,
        run_hybrid=not arguments.skip_hybrid,
        primary_model_name=arguments.primary_model,
        threshold_transport_mode=arguments.threshold_transport_mode,
        run_input_domain_masking=not arguments.skip_input_domain_masking,
        run_leave_one_domain_out_retraining=(
            not arguments.skip_leave_one_domain_out
        ),
        lodo_retrain_hybrid=not arguments.lodo_darn_only,
        lodo_bootstrap=arguments.lodo_bootstrap,
        lodo_save_models=arguments.save_lodo_models,
        run_eye_ear_adjusted_analysis=(
            not arguments.skip_eye_ear_adjusted_analysis
        ),
        run_eye_ear_feature_retraining=(
            not arguments.skip_eye_ear_feature_retraining
        ),
        run_tail_risk_strata_calibration=(
            not arguments.skip_tail_risk_strata_calibration
        ),
        run_false_negative_review=(
            not arguments.skip_false_negative_review
        ),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()




