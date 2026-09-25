#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUPPLEMENTARY MODEL-AUDIT ANALYSES — TEMPORAL-SAFE STRICT-OOF RUN
================================================

Final temporal-safe, strict-OOF run
-------------
/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/
darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653

Audit model
-------------
Hybrid DARN-XGB Blend, evaluated at the development-defined operating point
saved in test_model_metrics.csv.

This script generates
---------------------
1. Supplementary error-analysis figure:
   A. Confusion matrix.
   B. Predicted-risk distributions for TP/FP/FN/TN.
   C. Death capture versus alert burden.
   D. Descriptive clinical profiles of TP/FP/FN/TN.

2. Supplementary Figure S2 — subgroup operating-point audit:
   A. Specificity across prespecified subgroups.
   B. Positive predictive value across prespecified subgroups.

3. Supplementary Figure S3 — tail-focused calibration:
   A. Reliability plot using equal-frequency predicted-risk quantile bins
      with Wilson 95% confidence intervals for observed mortality.
   B. Number of encounters and deaths in each predicted-risk quantile bin.

4. Numerical tables:
   - Error-group counts and profiles.
   - Deidentified false-negative case review.
   - Death-capture/alert-burden summaries.
   - Subgroup specificity and PPV with bootstrap intervals.
   - Tail-calibration bin statistics.
   - Calibration intercept, slope, O/E ratio, and Brier score.
   - Manuscript-ready numerical text.

Important interpretation
------------------------
The subgroup and false-negative analyses are exploratory because the corrected
held-out test cohort contains only 47 deaths. They support model auditing but do not
establish demographic fairness, clinical safety, or external transportability.

The script intentionally does not reproduce decision-curve analysis because
that analysis is already included in the current main Figure 3.

Run in Jupyter
--------------
%run Reviewer_Supplementary_Analyses_strict_oof.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CURRENT-RUN SETTINGS
# =============================================================================

DEFAULT_RUN_DIR = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"
    "darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653"
)

DEFAULT_PIPELINE_MODULE = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Code/New/"
    "Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py"
)

DEFAULT_DATA_PATH = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"
    "Final_Datasets/df_final_mortality_2026-05-24_12-44_temporal_safe.csv"
)

DEFAULT_OUTPUT_SUBDIR = (
    "figures_revised_manuscript/"
    "reviewer_supplementary_strict_oof"
)

DEFAULT_AUDIT_MODEL = "Hybrid DARN-XGB Blend"
THRESHOLD_LABEL = "Development-defined operating point"

CAPTURE_MODEL_ORDER = [
    "Hybrid DARN-XGB Blend",
    "Hybrid DARN-XGB Stack",
    "XGBoost",
    "Uniform DARN",
    "MLP",
    "Logistic Regression",
    "ASA-only Logistic",
]

CMAP_NAME = "plasma"

N_BOOTSTRAP = 2000
N_JOBS = 8
RANDOM_STATE = 20260919

MIN_N_ANALYSIS = 30
MIN_EVENTS_FLAG = 5
N_CALIBRATION_BINS = 10

DISPLAY_ORDER = [
    ("Overall", "Overall"),
    ("Sex", "Female"),
    ("Sex", "Male"),
    ("Age", "<60"),
    ("Age", "60–69"),
    ("Age", "≥70"),
    ("ASA", "I–II"),
    ("ASA", "III"),
    ("ASA", "IV–V"),
    ("Urgency", "Elective"),
    ("Urgency", "Emergency"),
]

VARIABLE_ORDER = [
    "Overall",
    "Sex",
    "Age",
    "ASA",
    "Urgency",
]

ERROR_GROUP_ORDER = [
    "True positive",
    "False positive",
    "False negative",
    "True negative",
]

ERROR_GROUP_SHORT = {
    "True positive": "TP",
    "False positive": "FP",
    "False negative": "FN",
    "True negative": "TN",
}


# =============================================================================
# FILE AND GENERAL HELPERS
# =============================================================================

def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired current-run file not found:\n{path}\n"
            "Confirm the strict-OOF run directory and paths."
        )
    return path


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with open(require_file(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Mapping[str, Any], path: pathlib.Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        raise TypeError(f"Cannot serialize object of type {type(value)}")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=convert)


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    lower_map = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

        actual = lower_map.get(candidate.lower())
        if actual is not None:
            return actual

    return None


def normalize_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def load_pipeline_module(module_path: pathlib.Path):
    require_file(module_path)

    spec = importlib.util.spec_from_file_location(
        "mortality_darn_v6_reviewer_supplement_module",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import pipeline module: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_style() -> None:
    """Use enlarged publication typography consistent with the final figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "figure.titlesize": 19,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_all_formats(
    fig: plt.Figure,
    png_path: pathlib.Path,
) -> None:
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        png_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        png_path.with_suffix(".svg"),
        bbox_inches="tight",
    )

    try:
        fig.savefig(
            png_path.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception as exc:
        print("TIFF export skipped:", exc)


# =============================================================================
# FINAL TEMPORAL-SAFE STRICT-OOF HOLDOUT RECONSTRUCTION
# =============================================================================

def reconstruct_holdout(
    module,
    run_dir: pathlib.Path,
    data_path: pathlib.Path,
) -> Dict[str, Any]:
    config = load_json(
        run_dir / "config.json"
    )

    configured_data_path = pathlib.Path(
        str(
            config.get(
                "data_path",
                data_path,
            )
        )
    )

    if (
        data_path == DEFAULT_DATA_PATH
        and configured_data_path.exists()
    ):
        data_path = configured_data_path

    predictions = pd.read_csv(
        require_file(
            run_dir
            / "test_predictions_all_models.csv"
        ),
        low_memory=False,
    )

    if "y_true" not in predictions.columns:
        raise KeyError(
            "test_predictions_all_models.csv is missing 'y_true'."
        )

    source = pd.read_csv(
        require_file(data_path),
        low_memory=False,
    )

    target_column = str(
        config.get(
            "target_column",
            "mortality_30d",
        )
    )

    cohort, cohort_flow = (
        module.apply_reviewer_cohort_exclusions(
            source,
            target_column=target_column,
            exclude_asa6=bool(
                config.get(
                    "exclude_asa6",
                    True,
                )
            ),
        )
    )

    cohort = cohort.reset_index(
        drop=True
    )

    final_position_column = first_existing_column(
        predictions,
        [
            "final_cohort_row_position",
            "original_row_position",
        ],
    )

    if final_position_column is not None:
        positions = pd.to_numeric(
            predictions[
                final_position_column
            ],
            errors="raise",
        ).astype(int).to_numpy()

        if (
            np.any(positions < 0)
            or np.any(
                positions >= len(cohort)
            )
        ):
            raise IndexError(
                f"{final_position_column} contains positions outside "
                "the corrected cohort."
            )

        holdout = (
            cohort.iloc[
                positions
            ]
            .reset_index(
                drop=True
            )
        )

    elif "source_row_position" in predictions.columns:
        source_positions = pd.to_numeric(
            predictions[
                "source_row_position"
            ],
            errors="raise",
        ).astype(int).to_numpy()

        source_with_position = (
            source.reset_index(
                drop=False
            )
            .rename(
                columns={
                    "index": "source_row_position"
                }
            )
        )

        holdout = (
            source_with_position.set_index(
                "source_row_position"
            )
            .loc[
                source_positions
            ]
            .reset_index(
                drop=True
            )
        )

        positions = np.arange(
            len(holdout),
            dtype=int,
        )

    else:
        raise KeyError(
            "The test prediction file must contain one of "
            "'final_cohort_row_position', 'original_row_position', or "
            "'source_row_position'."
        )

    y_saved = pd.to_numeric(
        predictions["y_true"],
        errors="raise",
    ).astype(int).to_numpy()

    y_reconstructed = pd.to_numeric(
        holdout[
            target_column
        ],
        errors="raise",
    ).astype(int).to_numpy()

    if not np.array_equal(
        y_saved,
        y_reconstructed,
    ):
        raise RuntimeError(
            "Reconstructed test-set outcomes do not match saved predictions."
        )

    print("\n" + "=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF HOLDOUT RECONSTRUCTION")
    print("=" * 100)
    print(cohort_flow.to_string(index=False))
    print(f"Corrected cohort: {len(cohort):,}")
    print(
        f"Held-out test: N={len(y_saved):,}; deaths={int(y_saved.sum()):,}"
    )
    print(
        "Alignment column:",
        final_position_column
        if final_position_column is not None
        else "source_row_position",
    )

    return {
        "config": config,
        "data_path": data_path,
        "source": source,
        "cohort": cohort,
        "holdout": holdout,
        "predictions": predictions,
        "target_column": target_column,
        "y_true": y_saved,
        "holdout_positions": positions,
    }


# =============================================================================
# MODEL PROBABILITY AND THRESHOLD RESOLUTION
# =============================================================================

MODEL_ALIASES: Dict[str, List[str]] = {
    "Hybrid DARN-XGB Blend": [
        "hybriddarnxgbblend",
        "hybriddarnxgboostblend",
        "darnxgbblend",
        "darnxgboostblend",
        "hybridblend",
    ],
    "Hybrid DARN-XGB Stack": [
        "hybriddarnxgbstack",
        "hybriddarnxgbooststack",
        "hybridstack",
        "stackedhybrid",
    ],
    "XGBoost": [
        "xgboost",
    ],
    "Uniform DARN": [
        "uniformdarn",
        "darnuniformdomains",
        "uniformdomains",
    ],
    "MLP": [
        "standardmlp",
        "mlp",
    ],
    "Logistic Regression": [
        "logisticregression",
        "clinicallogistic",
        "fulllogistic",
    ],
    "ASA-only Logistic": [
        "asaonlylogistic",
        "asalogistic",
    ],
}



def match_model_row(
    metrics: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    if "model" not in metrics.columns:
        raise KeyError(
            "test_model_metrics.csv is missing the 'model' column."
        )

    target = normalize_name(
        model_name
    )

    normalized_models = metrics[
        "model"
    ].astype(str).map(
        normalize_name
    )

    exact = metrics.loc[
        normalized_models.eq(
            target
        )
    ]

    if not exact.empty:
        return exact.iloc[0]

    aliases = MODEL_ALIASES.get(
        model_name,
        [target],
    )

    for alias in aliases:
        matched = metrics.loc[
            normalized_models.str.contains(
                alias,
                regex=False,
            )
        ]

        if not matched.empty:
            return matched.iloc[0]

    raise KeyError(
        f"Model '{model_name}' was not found in test_model_metrics.csv.\n"
        f"Available models: {metrics['model'].astype(str).tolist()}"
    )


def resolve_probability_column(
    predictions: pd.DataFrame,
    model_name: str,
) -> str:
    probability_candidates = [
        str(column)
        for column in predictions.columns
        if (
            "prob" in str(column).lower()
            or "risk" in str(column).lower()
        )
    ]

    if not probability_candidates:
        raise KeyError(
            "No probability-like columns were found in the prediction file."
        )

    aliases = MODEL_ALIASES.get(
        model_name,
        [
            normalize_name(
                model_name
            )
        ],
    )

    scored: List[
        Tuple[float, str]
    ] = []

    for column in probability_candidates:
        normalized_column = normalize_name(
            column
        )

        score = 0.0

        for alias in aliases:
            if alias in normalized_column:
                score += 20.0 + len(alias) / 100.0

        if "calibrated" in normalized_column:
            score += 5.0

        if "probability" in normalized_column:
            score += 2.0

        if normalized_column.startswith("prob"):
            score += 1.0

        # Prevent selecting plain XGBoost for the hybrid model.
        if (
            model_name == "Hybrid DARN-XGB Blend"
            and "hybrid" not in normalized_column
            and "blend" not in normalized_column
        ):
            score -= 10.0

        scored.append(
            (
                score,
                column,
            )
        )

    scored.sort(
        reverse=True
    )

    best_score, best_column = scored[0]

    if best_score <= 0:
        raise KeyError(
            f"Could not resolve a probability column for '{model_name}'.\n"
            f"Available probability-like columns: {probability_candidates}"
        )

    return best_column


def load_audit_model(
    run_dir: pathlib.Path,
    predictions: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    metrics = pd.read_csv(
        require_file(
            run_dir
            / "test_model_metrics.csv"
        )
    )

    model_row = match_model_row(
        metrics,
        model_name,
    )

    threshold_column = first_existing_column(
        metrics,
        [
            "Threshold",
            "threshold",
        ],
    )

    if threshold_column is None:
        raise KeyError(
            "test_model_metrics.csv has no threshold column."
        )

    threshold = float(
        model_row[
            threshold_column
        ]
    )

    probability_column = (
        resolve_probability_column(
            predictions,
            model_name,
        )
    )

    probability = pd.to_numeric(
        predictions[
            probability_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if np.any(
        ~np.isfinite(
            probability
        )
    ):
        raise ValueError(
            "Audit-model probabilities contain non-finite values."
        )

    return {
        "metrics_table": metrics,
        "model_row": model_row,
        "model_name": model_name,
        "probability_column": probability_column,
        "probability": probability,
        "threshold": threshold,
    }


def resolve_available_model_probabilities(
    predictions: pd.DataFrame,
    model_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    output: Dict[
        str,
        np.ndarray,
    ] = {}

    for model_name in model_names:
        try:
            column = resolve_probability_column(
                predictions,
                model_name,
            )

            probability = pd.to_numeric(
                predictions[
                    column
                ],
                errors="raise",
            ).to_numpy(
                dtype=float
            )

            if np.all(
                np.isfinite(
                    probability
                )
            ):
                output[
                    model_name
                ] = probability

                print(
                    f"Capture curve model: {model_name} <- {column}"
                )

        except Exception as exc:
            print(
                f"Skipping capture curve for {model_name}: {exc}"
            )

    return output


# =============================================================================
# SUBGROUP NORMALIZATION
# =============================================================================

def normalize_sex(
    series: pd.Series,
) -> pd.Series:
    raw = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    output = pd.Series(
        "Missing/Other",
        index=series.index,
        dtype=object,
    )

    output.loc[
        raw.isin(
            [
                "0",
                "false",
                "f",
                "female",
                "woman",
                "women",
            ]
        )
        | raw.str.contains(
            "female",
            na=False,
        )
    ] = "Female"

    output.loc[
        raw.isin(
            [
                "1",
                "true",
                "m",
                "male",
                "man",
                "men",
            ]
        )
        | (
            raw.str.contains(
                "male",
                na=False,
            )
            & ~raw.str.contains(
                "female",
                na=False,
            )
        )
    ] = "Male"

    return output


def normalize_emergency(
    series: pd.Series,
) -> pd.Series:
    raw = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    output = pd.Series(
        "Missing/Other",
        index=series.index,
        dtype=object,
    )

    output.loc[
        numeric.eq(0)
        | raw.str.contains(
            "elect",
            na=False,
        )
        | raw.str.contains(
            "sched",
            na=False,
        )
    ] = "Elective"

    output.loc[
        numeric.eq(1)
        | raw.str.contains(
            "emerg",
            na=False,
        )
        | raw.str.contains(
            "urgent",
            na=False,
        )
    ] = "Emergency"

    return output


def parse_asa_numeric(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    roman_map = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
    }

    def parse_one(value: Any) -> float:
        if pd.isna(value):
            return np.nan

        text = str(value).strip().lower()

        match = re.search(
            r"\b(vi|iv|v|iii|ii|i)\b",
            text,
        )

        if match:
            return float(
                roman_map[
                    match.group(1)
                ]
            )

        match = re.search(
            r"([1-6])",
            text,
        )

        if match:
            return float(
                match.group(1)
            )

        return np.nan

    missing = numeric.isna()

    numeric.loc[
        missing
    ] = (
        series.loc[
            missing
        ]
        .map(
            parse_one
        )
        .astype(float)
    )

    return numeric.astype(float)


def derive_subgroups(
    holdout: pd.DataFrame,
) -> Dict[str, pd.Series]:
    groups: Dict[
        str,
        pd.Series,
    ] = {}

    sex_column = first_existing_column(
        holdout,
        [
            "sex",
            "gender",
        ],
    )
    age_column = first_existing_column(
        holdout,
        [
            "age",
        ],
    )

    asa_column = first_existing_column(
        holdout,
        [
            "asa",
            "asa_class",
            "asa_status",
            "asa_ps",
        ],
    )

    urgency_column = first_existing_column(
        holdout,
        [
            "emop",
            "emergency",
            "emergency_status",
            "urgency",
            "urgent",
        ],
    )

    if sex_column is not None:
        groups["Sex"] = normalize_sex(
            holdout[
                sex_column
            ]
        )

    if age_column is not None:
        age = clean_numeric(
            holdout[
                age_column
            ]
        )

        age_group = pd.Series(
            "Missing/Other",
            index=holdout.index,
            dtype=object,
        )

        age_group.loc[
            age.lt(60)
        ] = "<60"

        age_group.loc[
            age.ge(60)
            & age.lt(70)
        ] = "60–69"

        age_group.loc[
            age.ge(70)
        ] = "≥70"

        groups["Age"] = age_group

    if asa_column is not None:
        asa = parse_asa_numeric(
            holdout[
                asa_column
            ]
        )

        if asa.eq(6).any():
            raise RuntimeError(
                "ASA 6 encounters remain in the corrected holdout."
            )

        asa_group = pd.Series(
            "Missing/Other",
            index=holdout.index,
            dtype=object,
        )

        asa_group.loc[
            asa.isin(
                [1, 2]
            )
        ] = "I–II"

        asa_group.loc[
            asa.eq(3)
        ] = "III"

        asa_group.loc[
            asa.isin(
                [4, 5]
            )
        ] = "IV–V"

        groups["ASA"] = asa_group

    if urgency_column is not None:
        groups["Urgency"] = (
            normalize_emergency(
                holdout[
                    urgency_column
                ]
            )
        )

    missing = {
        "Sex",
        "Age",
        "ASA",
        "Urgency",
    } - set(groups)

    if missing:
        print(
            "Warning: unavailable subgroup variables:",
            sorted(
                missing
            ),
        )

    return groups


# =============================================================================
# ERROR GROUPS AND CLINICAL PROFILES
# =============================================================================

def assign_error_groups(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    prediction = (
        probability
        >= threshold
    ).astype(int)

    return np.select(
        [
            (y_true == 1)
            & (prediction == 1),

            (y_true == 0)
            & (prediction == 1),

            (y_true == 1)
            & (prediction == 0),

            (y_true == 0)
            & (prediction == 0),
        ],
        ERROR_GROUP_ORDER,
        default="Unknown",
    )


def build_error_profile(
    holdout: pd.DataFrame,
    error_groups: np.ndarray,
    probability: np.ndarray,
    subgroups: Mapping[str, pd.Series],
) -> pd.DataFrame:
    age_group = subgroups.get(
        "Age",
        pd.Series(
            "Missing/Other",
            index=holdout.index,
        ),
    )

    asa_group = subgroups.get(
        "ASA",
        pd.Series(
            "Missing/Other",
            index=holdout.index,
        ),
    )

    urgency_group = subgroups.get(
        "Urgency",
        pd.Series(
            "Missing/Other",
            index=holdout.index,
        ),
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for group in ERROR_GROUP_ORDER:
        mask = (
            error_groups
            == group
        )

        n_group = int(
            mask.sum()
        )

        rows.append(
            {
                "error_group": group,
                "n": n_group,
                "Age ≥70 (%)": (
                    100.0
                    * float(
                        age_group.loc[
                            mask
                        ].eq(
                            "≥70"
                        ).mean()
                    )
                    if n_group
                    else np.nan
                ),
                "ASA III–V (%)": (
                    100.0
                    * float(
                        asa_group.loc[
                            mask
                        ].isin(
                            [
                                "III",
                                "IV–V",
                            ]
                        ).mean()
                    )
                    if n_group
                    else np.nan
                ),
                "Emergency (%)": (
                    100.0
                    * float(
                        urgency_group.loc[
                            mask
                        ].eq(
                            "Emergency"
                        ).mean()
                    )
                    if n_group
                    else np.nan
                ),
                "Median predicted risk": (
                    float(
                        np.median(
                            probability[
                                mask
                            ]
                        )
                    )
                    if n_group
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_false_negative_review(
    holdout: pd.DataFrame,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    error_groups: np.ndarray,
    subgroups: Mapping[str, pd.Series],
) -> pd.DataFrame:
    false_negative_positions = np.where(
        error_groups
        == "False negative"
    )[0]

    probability_percentile = pd.Series(
        probability
    ).rank(
        method="average",
        pct=True,
    ).to_numpy()

    candidate_columns: Dict[
        str,
        Sequence[str],
    ] = {
        "age": [
            "age",
        ],
        "sex_raw": [
            "sex",
            "gender",
        ],
        "asa_raw": [
            "asa",
            "asa_class",
            "asa_status",
            "asa_ps",
        ],
        "urgency_raw": [
            "emop",
            "emergency",
            "emergency_status",
            "urgency",
        ],
        "department": [
            "department",
            "surgical_department",
            "surgery_department",
            "department_name",
            "op_department",
            "optype",
            "surgery_type",
            "surgical_service",
        ],
        "surgery_duration": [
            "surgery_duration",
        ],
        "anesthesia_duration": [
            "anesthesia_duration",
        ],
        "or_duration": [
            "or_duration",
        ],
        "blood_loss": [
            "intraop_mean_ebl",
            "mean_ebl",
            "ebl",
        ],
        "fibrinogen": [
            "preop_fibrinogen",
            "fibrinogen",
        ],
        "platelets": [
            "preop_platelet",
            "preop_platelets",
            "platelet",
            "platelets",
        ],
        "glucose": [
            "preop_glucose",
            "glucose",
        ],
    }

    resolved_columns: Dict[
        str,
        str,
    ] = {}

    for output_name, candidates in candidate_columns.items():
        column = first_existing_column(
            holdout,
            candidates,
        )

        if column is not None:
            resolved_columns[
                output_name
            ] = column

    diagnosis_candidates = [
        column
        for column in holdout.columns
        if str(column).upper().startswith(
            "DIAG_"
        )
    ]

    review_rows: List[
        Dict[str, Any]
    ] = []

    for position in false_negative_positions:
        row: Dict[
            str,
            Any,
        ] = {
            "holdout_row_position": int(
                position
            ),
            "predicted_probability": float(
                probability[
                    position
                ]
            ),
            "threshold": float(
                threshold
            ),
            "distance_below_threshold": float(
                threshold
                - probability[
                    position
                ]
            ),
            "predicted_risk_percentile": float(
                probability_percentile[
                    position
                ]
            ),
            "age_group": (
                str(
                    subgroups[
                        "Age"
                    ].iloc[
                        position
                    ]
                )
                if "Age" in subgroups
                else "Unavailable"
            ),
            "sex_group": (
                str(
                    subgroups[
                        "Sex"
                    ].iloc[
                        position
                    ]
                )
                if "Sex" in subgroups
                else "Unavailable"
            ),
            "asa_group": (
                str(
                    subgroups[
                        "ASA"
                    ].iloc[
                        position
                    ]
                )
                if "ASA" in subgroups
                else "Unavailable"
            ),
            "urgency_group": (
                str(
                    subgroups[
                        "Urgency"
                    ].iloc[
                        position
                    ]
                )
                if "Urgency" in subgroups
                else "Unavailable"
            ),
            "raw_missing_count": int(
                holdout.iloc[
                    position
                ].isna().sum()
            ),
            "raw_missing_fraction": float(
                holdout.iloc[
                    position
                ].isna().mean()
            ),
        }

        for output_name, column in resolved_columns.items():
            row[
                output_name
            ] = holdout.iloc[
                position
            ][
                column
            ]

        present_diagnoses: List[
            str
        ] = []

        for diagnosis_column in diagnosis_candidates:
            value = pd.to_numeric(
                pd.Series(
                    [
                        holdout.iloc[
                            position
                        ][
                            diagnosis_column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if (
                np.isfinite(
                    value
                )
                and value > 0
            ):
                present_diagnoses.append(
                    str(
                        diagnosis_column
                    )
                )

        row[
            "present_diagnoses"
        ] = " | ".join(
            present_diagnoses[
                :20
            ]
        )

        review_rows.append(
            row
        )

    return pd.DataFrame(
        review_rows
    )


# =============================================================================
# DEATH-CAPTURE / ALERT-BURDEN CURVES
# =============================================================================

def build_capture_curves(
    y_true: np.ndarray,
    model_probabilities: Mapping[str, np.ndarray],
    audit_threshold: float,
    audit_model: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(
        y_true
    )
    total_events = int(
        y_true.sum()
    )

    curve_rows: List[
        Dict[str, Any]
    ] = []

    summary_rows: List[
        Dict[str, Any]
    ] = []

    alert_percent_targets = [
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ]

    for model_name, probability in model_probabilities.items():
        order = np.argsort(
            -probability
        )

        ordered_y = y_true[
            order
        ]

        cumulative_events = np.cumsum(
            ordered_y
        )

        alerted_percent = (
            100.0
            * (
                np.arange(
                    n
                )
                + 1
            )
            / n
        )

        captured_percent = (
            100.0
            * cumulative_events
            / total_events
        )

        # Downsample only for the saved plotting curve.
        keep_indices = np.unique(
            np.linspace(
                0,
                n - 1,
                min(
                    2000,
                    n,
                ),
            ).astype(int)
        )

        for index in keep_indices:
            curve_rows.append(
                {
                    "model": model_name,
                    "alerted_percent": float(
                        alerted_percent[
                            index
                        ]
                    ),
                    "captured_deaths_percent": float(
                        captured_percent[
                            index
                        ]
                    ),
                    "captured_deaths_n": int(
                        cumulative_events[
                            index
                        ]
                    ),
                    "alerted_n": int(
                        index + 1
                    ),
                }
            )

        for target in alert_percent_targets:
            alerted_n = max(
                1,
                int(
                    np.ceil(
                        target
                        / 100.0
                        * n
                    )
                ),
            )

            captured_n = int(
                cumulative_events[
                    alerted_n - 1
                ]
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "operating_point": (
                        f"Top {target:g}%"
                    ),
                    "alerted_n": alerted_n,
                    "alerted_percent": (
                        100.0
                        * alerted_n
                        / n
                    ),
                    "captured_deaths_n": captured_n,
                    "captured_deaths_percent": (
                        100.0
                        * captured_n
                        / total_events
                    ),
                }
            )

        if model_name == audit_model:
            predicted_positive = (
                probability
                >= audit_threshold
            )

            alerted_n = int(
                predicted_positive.sum()
            )

            captured_n = int(
                y_true[
                    predicted_positive
                ].sum()
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "operating_point": (
                        THRESHOLD_LABEL
                    ),
                    "alerted_n": alerted_n,
                    "alerted_percent": (
                        100.0
                        * alerted_n
                        / n
                    ),
                    "captured_deaths_n": captured_n,
                    "captured_deaths_percent": (
                        100.0
                        * captured_n
                        / total_events
                    ),
                }
            )

    return (
        pd.DataFrame(
            curve_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# =============================================================================
# OPERATING-POINT METRICS AND BOOTSTRAP
# =============================================================================

def operating_point_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    prediction = (
        probability
        >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (prediction == 1)
            & (y_true == 1)
        )
    )
    fp = int(
        np.sum(
            (prediction == 1)
            & (y_true == 0)
        )
    )
    tn = int(
        np.sum(
            (prediction == 0)
            & (y_true == 0)
        )
    )
    fn = int(
        np.sum(
            (prediction == 0)
            & (y_true == 1)
        )
    )

    return {
        "n": int(
            len(
                y_true
            )
        ),
        "events": int(
            y_true.sum()
        ),
        "prevalence": float(
            y_true.mean()
        ),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": safe_divide(
            tp,
            tp + fn,
        ),
        "specificity": safe_divide(
            tn,
            tn + fp,
        ),
        "PPV": safe_divide(
            tp,
            tp + fp,
        ),
        "NPV": safe_divide(
            tn,
            tn + fn,
        ),
        "FPR": safe_divide(
            fp,
            fp + tn,
        ),
        "FNR": safe_divide(
            fn,
            fn + tp,
        ),
        "alerts_per_1000": float(
            1000.0
            * (
                tp + fp
            )
            / len(
                y_true
            )
        ),
    }


def percentile_interval(
    values: Sequence[float],
) -> Tuple[float, float]:
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if len(array) == 0:
        return np.nan, np.nan

    return (
        float(
            np.quantile(
                array,
                0.025,
            )
        ),
        float(
            np.quantile(
                array,
                0.975,
            )
        ),
    )


def bootstrap_operating_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    seed: int,
    n_jobs: int,
) -> Dict[str, Tuple[float, float, int]]:
    metric_names = [
        "specificity",
        "PPV",
        "sensitivity",
        "FPR",
        "NPV",
        "FNR",
    ]

    n = len(
        y_true
    )

    positive_indices = np.where(
        y_true == 1
    )[0]
    negative_indices = np.where(
        y_true == 0
    )[0]

    seed_sequence = np.random.SeedSequence(
        seed
    )

    replicate_seeds = [
        int(
            child.generate_state(
                1
            )[0]
        )
        for child in seed_sequence.spawn(
            n_bootstrap
        )
    ]

    def one_replicate(
        replicate_seed: int,
    ) -> Dict[str, float]:
        rng = np.random.default_rng(
            replicate_seed
        )

        if (
            len(positive_indices) > 0
            and len(negative_indices) > 0
        ):
            sampled_positive = rng.choice(
                positive_indices,
                size=len(positive_indices),
                replace=True,
            )
            sampled_negative = rng.choice(
                negative_indices,
                size=len(negative_indices),
                replace=True,
            )
            indices = np.concatenate(
                [sampled_positive, sampled_negative]
            )
            rng.shuffle(indices)
        else:
            indices = rng.integers(
                0,
                n,
                size=n,
            )

        return operating_point_metrics(
            y_true[
                indices
            ],
            probability[
                indices
            ],
            threshold,
        )

    results = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        verbose=0,
    )(
        delayed(
            one_replicate
        )(
            replicate_seed
        )
        for replicate_seed in replicate_seeds
    )

    output: Dict[
        str,
        Tuple[
            float,
            float,
            int,
        ],
    ] = {}

    for metric in metric_names:
        values = np.asarray(
            [
                result[
                    metric
                ]
                for result in results
            ],
            dtype=float,
        )

        lower, upper = percentile_interval(
            values
        )

        output[
            metric
        ] = (
            lower,
            upper,
            int(
                np.isfinite(
                    values
                ).sum()
            ),
        )

    return output


def build_subgroup_operating_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    subgroups: Mapping[str, pd.Series],
    *,
    n_bootstrap: int,
    n_jobs: int,
) -> pd.DataFrame:
    definitions: List[
        Dict[str, Any]
    ] = [
        {
            "subgroup_variable": "Overall",
            "subgroup_level": "Overall",
            "mask": np.ones(
                len(
                    y_true
                ),
                dtype=bool,
            ),
        }
    ]

    for variable, values in subgroups.items():
        values = values.reset_index(
            drop=True
        )

        for level in pd.unique(
            values
        ):
            level_text = str(
                level
            )
            if level_text == "Missing/Other":
                continue

            definitions.append(
                {
                    "subgroup_variable": variable,
                    "subgroup_level": level_text,
                    "mask": values.eq(
                        level_text
                    ).to_numpy(),
                }
            )

    rows: List[
        Dict[str, Any]
    ] = []

    print("\n" + "=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF SUBGROUP SPECIFICITY/PPV BOOTSTRAP")
    print("=" * 100)

    for index, definition in enumerate(
        definitions
    ):
        mask = np.asarray(
            definition[
                "mask"
            ],
            dtype=bool,
        )

        y_group = y_true[
            mask
        ]

        p_group = probability[
            mask
        ]

        if len(
            y_group
        ) < MIN_N_ANALYSIS:
            continue

        print(
            f"[{index + 1:02d}/{len(definitions):02d}] "
            f"{definition['subgroup_variable']}: "
            f"{definition['subgroup_level']} | "
            f"N={len(y_group):,}; deaths={int(y_group.sum())}"
        )

        point = operating_point_metrics(
            y_group,
            p_group,
            threshold,
        )

        intervals = bootstrap_operating_metrics(
            y_group,
            p_group,
            threshold,
            n_bootstrap=n_bootstrap,
            seed=(
                RANDOM_STATE
                + 1009
                * (
                    index + 1
                )
            ),
            n_jobs=n_jobs,
        )

        row: Dict[
            str,
            Any,
        ] = {
            "subgroup_variable": definition[
                "subgroup_variable"
            ],
            "subgroup_level": definition[
                "subgroup_level"
            ],
            **point,
            "low_event_flag": int(
                point[
                    "events"
                ]
                < MIN_EVENTS_FLAG
            ),
        }

        for metric, (
            lower,
            upper,
            valid,
        ) in intervals.items():
            row[
                f"{metric}_ci_lower_95"
            ] = lower
            row[
                f"{metric}_ci_upper_95"
            ] = upper
            row[
                f"{metric}_valid_bootstraps"
            ] = valid

        rows.append(
            row
        )

    table = pd.DataFrame(
        rows
    )

    order_map = {
        pair: order
        for order, pair in enumerate(
            DISPLAY_ORDER
        )
    }

    table[
        "display_order"
    ] = [
        order_map.get(
            (
                row.subgroup_variable,
                row.subgroup_level,
            ),
            999,
        )
        for row in table.itertuples()
    ]

    table[
        "display_label"
    ] = table.apply(
        lambda row: (
            "Overall"
            if row[
                "subgroup_variable"
            ]
            == "Overall"
            else (
                f"{row['subgroup_variable']}: "
                f"{row['subgroup_level']}"
            )
        ),
        axis=1,
    )

    table["bootstrap_method"] = (
        "Patient-level stratified bootstrap within subgroup"
    )
    table["bootstrap_replicates"] = int(
        n_bootstrap
    )
    table["operating_threshold_source"] = (
        "Development out-of-fold predictions"
    )

    table[
        "display_label_with_counts"
    ] = (
        table[
            "display_label"
        ]
        + table.apply(
            lambda row: (
                f"  (N={int(row['n']):,}; "
                f"deaths={int(row['events'])})"
            ),
            axis=1,
        )
    )

    return table.sort_values(
        [
            "display_order",
            "subgroup_variable",
            "subgroup_level",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def wilson_interval(
    events: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan

    proportion = events / n
    denominator = (
        1.0
        + z**2
        / n
    )

    center = (
        proportion
        + z**2
        / (
            2.0
            * n
        )
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            proportion
            * (
                1.0
                - proportion
            )
            / n
            + z**2
            / (
                4.0
                * n**2
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center
            - half_width,
        ),
        min(
            1.0,
            center
            + half_width,
        ),
    )


def fit_calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    epsilon = 1e-9

    probability = np.clip(
        probability,
        epsilon,
        1.0
        - epsilon,
    )

    logit_probability = np.log(
        probability
        / (
            1.0
            - probability
        )
    )

    def negative_log_likelihood(
        parameters: np.ndarray,
    ) -> float:
        intercept, slope = parameters

        linear_predictor = (
            intercept
            + slope
            * logit_probability
        )

        # Stable Bernoulli negative log-likelihood.
        return float(
            np.sum(
                np.logaddexp(
                    0.0,
                    linear_predictor,
                )
                - y_true
                * linear_predictor
            )
        )

    result = minimize(
        negative_log_likelihood,
        x0=np.asarray(
            [
                0.0,
                1.0,
            ],
            dtype=float,
        ),
        method="BFGS",
    )

    intercept = float(
        result.x[0]
    )
    slope = float(
        result.x[1]
    )

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "optimization_success": bool(
            result.success
        ),
        "optimization_message": str(
            result.message
        ),
    }


def build_tail_calibration_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    # Equal-frequency quantile bins provide approximately equal numbers of holdout encounters per bin and stabilize descriptive tail estimates.
    ranks = pd.Series(
        probability
    ).rank(
        method="first",
    )

    bin_labels = pd.qcut(
        ranks,
        q=min(
            n_bins,
            len(
                ranks
            ),
        ),
        labels=False,
        duplicates="drop",
    )

    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "probability": probability,
            "risk_bin": (
                bin_labels.astype(int)
                + 1
            ),
        }
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for risk_bin, subset in frame.groupby(
        "risk_bin",
        observed=True,
    ):
        n = int(
            len(
                subset
            )
        )
        events = int(
            subset[
                "y_true"
            ].sum()
        )

        lower, upper = wilson_interval(
            events,
            n,
        )

        rows.append(
            {
                "risk_bin": int(
                    risk_bin
                ),
                "n": n,
                "events": events,
                "mean_predicted_risk": float(
                    subset[
                        "probability"
                    ].mean()
                ),
                "median_predicted_risk": float(
                    subset[
                        "probability"
                    ].median()
                ),
                "minimum_predicted_risk": float(
                    subset[
                        "probability"
                    ].min()
                ),
                "maximum_predicted_risk": float(
                    subset[
                        "probability"
                    ].max()
                ),
                "observed_mortality": float(
                    subset[
                        "y_true"
                    ].mean()
                ),
                "observed_ci_lower_95": lower,
                "observed_ci_upper_95": upper,
            }
        )

    table = pd.DataFrame(
        rows
    ).sort_values(
        "risk_bin"
    ).reset_index(
        drop=True
    )

    table["binning_method"] = (
        "Equal-frequency quantile bins based on ranked predicted probabilities"
    )
    table["observed_mortality_interval_method"] = (
        "Wilson score 95% confidence interval"
    )
    table["horizontal_error_bar_definition"] = (
        "Minimum-to-maximum predicted probability within bin"
    )

    return table


# =============================================================================
# FIGURE 1 — ERROR ANALYSIS
# =============================================================================

def create_error_figure(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    error_groups: np.ndarray,
    error_profile: pd.DataFrame,
    capture_curve: pd.DataFrame,
    capture_summary: pd.DataFrame,
    audit_model: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    configure_style()

    cmap = cm.get_cmap(
        CMAP_NAME
    )

    error_colors = {
        group: cmap(
            value
        )
        for group, value in zip(
            ERROR_GROUP_ORDER,
            [
                0.92,
                0.68,
                0.38,
                0.10,
            ],
        )
    }

    counts = (
        pd.Series(
            error_groups
        )
        .value_counts()
        .reindex(
            ERROR_GROUP_ORDER,
            fill_value=0,
        )
    )

    tp = int(
        counts[
            "True positive"
        ]
    )
    fp = int(
        counts[
            "False positive"
        ]
    )
    fn = int(
        counts[
            "False negative"
        ]
    )
    tn = int(
        counts[
            "True negative"
        ]
    )

    confusion = np.asarray(
        [
            [
                tp,
                fn,
            ],
            [
                fp,
                tn,
            ],
        ],
        dtype=int,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            19,
            14.5,
        ),
    )

    (
        (
            axis_a,
            axis_b,
        ),
        (
            axis_c,
            axis_d,
        ),
    ) = axes

    fig.suptitle(
        "Error patterns and operating characteristics of the hybrid DARN–XGBoost blend",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )

    # Test-set size and threshold provenance are reported in the caption.

    # Panel A — Confusion matrix
    image = axis_a.imshow(
        confusion,
        cmap=CMAP_NAME,
        aspect="equal",
    )

    axis_a.set_xticks(
        [
            0,
            1,
        ]
    )
    axis_a.set_yticks(
        [
            0,
            1,
        ]
    )

    axis_a.set_xticklabels(
        [
            "Predicted death",
            "Predicted survival",
        ]
    )
    axis_a.set_yticklabels(
        [
            "Observed death",
            "Observed survival",
        ]
    )

    row_totals = confusion.sum(
        axis=1
    )

    for row_index in range(
        2
    ):
        for column_index in range(
            2
        ):
            count = int(
                confusion[
                    row_index,
                    column_index,
                ]
            )

            row_percent = (
                100.0
                * count
                / row_totals[
                    row_index
                ]
                if row_totals[
                    row_index
                ]
                > 0
                else np.nan
            )

            # Choose annotation color from the actual cell background.
            # Plasma is dark for low counts and bright for high counts, so
            # low-count cells need white text while bright cells need black.
            rgba = image.cmap(image.norm(count))
            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )
            text_color = "black" if luminance > 0.55 else "white"

            axis_a.text(
                column_index,
                row_index,
                (
                    f"{count:,}\n"
                    f"{row_percent:.1f}% of row"
                ),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=text_color,
            )

    axis_a.set_title(
        "A. Classification outcomes",
        fontweight="bold",
        loc="left",
    )

    colorbar = fig.colorbar(
        image,
        ax=axis_a,
        fraction=0.045,
        pad=0.03,
    )

    colorbar.set_label(
        "Number of encounters"
    )

    # Panel B — Predicted-risk distributions
    box_data = [
        100.0
        * probability[
            error_groups
            == group
        ]
        for group in ERROR_GROUP_ORDER
    ]

    boxplot = axis_b.boxplot(
        box_data,
        tick_labels=[
            ERROR_GROUP_SHORT[
                group
            ]
            for group in ERROR_GROUP_ORDER
        ],
        patch_artist=True,
        showfliers=False,
        widths=0.62,
    )

    for patch, group in zip(
        boxplot[
            "boxes"
        ],
        ERROR_GROUP_ORDER,
    ):
        patch.set_facecolor(
            error_colors[
                group
            ]
        )
        patch.set_alpha(
            0.85
        )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for position, (
        group,
        values,
    ) in enumerate(
        zip(
            ERROR_GROUP_ORDER,
            box_data,
        ),
        start=1,
    ):
        finite = np.asarray(
            values,
            dtype=float,
        )

        finite = finite[
            np.isfinite(
                finite
            )
        ]

        if len(
            finite
        ) > 700:
            finite = rng.choice(
                finite,
                size=700,
                replace=False,
            )

        jitter = rng.normal(
            position,
            0.055,
            size=len(
                finite
            ),
        )

        axis_b.scatter(
            jitter,
            np.maximum(
                finite,
                1e-7,
            ),
            s=7,
            alpha=0.23,
            color=error_colors[
                group
            ],
            edgecolors="none",
            rasterized=True,
        )

    axis_b.axhline(
        100.0
        * threshold,
        linestyle="--",
        linewidth=1.2,
        color="black",
        label=THRESHOLD_LABEL,
    )

    axis_b.set_yscale(
        "log"
    )
    axis_b.set_xlabel(
        "Classification outcome"
    )
    axis_b.set_ylabel(
        "Predicted mortality risk (%) — log scale"
    )
    axis_b.set_title(
        "B. Predicted-risk distributions",
        fontweight="bold",
        loc="left",
    )
    axis_b.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )
    axis_b.legend(
        frameon=False,
        fontsize=11.5,
    )

    # Panel C — Death capture versus alert burden
    model_names = (
        capture_curve[
            "model"
        ]
        .drop_duplicates()
        .tolist()
    )

    model_colors = {
        model_name: cmap(
            value
        )
        for model_name, value in zip(
            model_names,
            np.linspace(
                0.08,
                0.92,
                max(
                    len(
                        model_names
                    ),
                    1,
                ),
            ),
        )
    }

    for model_name in model_names:
        subset = capture_curve.loc[
            capture_curve[
                "model"
            ].eq(
                model_name
            )
        ]

        axis_c.plot(
            subset[
                "alerted_percent"
            ],
            subset[
                "captured_deaths_percent"
            ],
            linewidth=(
                2.8
                if model_name
                == audit_model
                else 1.8
            ),
            color=model_colors[
                model_name
            ],
            label=model_name,
        )

    for percentage in [
        1.0,
        2.0,
        5.0,
    ]:
        axis_c.axvline(
            percentage,
            color="lightgray",
            linestyle=":",
            linewidth=0.9,
        )

    primary_operating = (
        capture_summary.loc[
            capture_summary[
                "model"
            ].eq(
                audit_model
            )
            & capture_summary[
                "operating_point"
            ].eq(
                THRESHOLD_LABEL
            )
        ]
    )

    if not primary_operating.empty:
        row = primary_operating.iloc[
            0
        ]

        axis_c.scatter(
            [
                row[
                    "alerted_percent"
                ]
            ],
            [
                row[
                    "captured_deaths_percent"
                ]
            ],
            s=85,
            facecolor="white",
            edgecolor=model_colors[
                audit_model
            ],
            linewidth=1.8,
            zorder=5,
        )

        axis_c.annotate(
            (
                f"Selected threshold\n"
                f"{row['captured_deaths_n']:.0f}/"
                f"{int(y_true.sum())} deaths; "
                f"{row['alerted_percent']:.1f}% alerted"
            ),
            xy=(
                row[
                    "alerted_percent"
                ],
                row[
                    "captured_deaths_percent"
                ],
            ),
            xytext=(
                min(
                    row[
                        "alerted_percent"
                    ]
                    + 2.0,
                    16.0,
                ),
                max(
                    row[
                        "captured_deaths_percent"
                    ]
                    - 22.0,
                    10.0,
                ),
            ),
            fontsize=11.0,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 0.8,
                "color": model_colors[
                    audit_model
                ],
            },
        )

    axis_c.set_xlim(
        0.0,
        20.0,
    )
    axis_c.set_ylim(
        0.0,
        102.0,
    )
    axis_c.set_xlabel(
        "Held-out test encounters flagged (%)"
    )
    axis_c.set_ylabel(
        "Deaths captured (%)"
    )
    axis_c.set_title(
        "C. Death capture versus alert burden",
        fontweight="bold",
        loc="left",
    )
    axis_c.grid(
        linestyle="--",
        alpha=0.18,
    )
    axis_c.legend(
        frameon=False,
        fontsize=11.0,
        loc="lower right",
    )

    # Panel D — Clinical profiles
    profile_metrics = [
        "Age ≥70 (%)",
        "ASA III–V (%)",
        "Emergency (%)",
    ]

    profile_indexed = (
        error_profile.set_index(
            "error_group"
        )
        .reindex(
            ERROR_GROUP_ORDER
        )
    )

    x_positions = np.arange(
        len(
            ERROR_GROUP_ORDER
        )
    )

    width = 0.24

    metric_colors = [
        cmap(
            0.20
        ),
        cmap(
            0.50
        ),
        cmap(
            0.80
        ),
    ]

    for metric_index, (
        metric,        color,
    ) in enumerate(
        zip(
            profile_metrics,
            metric_colors,
        )
    ):
        values = profile_indexed[
            metric
        ].to_numpy(
            dtype=float
        )

        axis_d.bar(
            x_positions
            + (
                metric_index
                - 1
            )
            * width,
            values,
            width=width,
            label=metric,
            color=color,
            edgecolor="none",
        )

    axis_d.set_xticks(
        x_positions
    )
    axis_d.set_xticklabels(
        [
            ERROR_GROUP_SHORT[
                group
            ]
            for group in ERROR_GROUP_ORDER
        ]
    )
    axis_d.set_ylim(
        0.0,
        100.0,
    )
    axis_d.set_xlabel(
        "Classification outcome"
    )
    axis_d.set_ylabel(
        "Encounters in group (%)"
    )
    axis_d.set_title(
        "D. Descriptive clinical profiles",
        fontweight="bold",
        loc="left",
    )
    axis_d.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )
    axis_d.legend(
        frameon=False,
        fontsize=11.5,
    )

    fig.text(
        0.05,
        0.018,
        (
            "TP=true positive; FP=false positive; FN=false negative; "
            "TN=true negative. Error-group comparisons are descriptive. "
        ),
        ha="left",
        va="bottom",
        fontsize=11.0,
        color="dimgray",
    )

    fig.tight_layout(
        rect=(
            0.03,
            0.055,
            0.99,
            0.955,
        ),
        h_pad=2.8,
        w_pad=2.6,
    )

    output_path = (
        output_dir
        / "FigS_Error_Analysis_StrictOOF.png"
    )

    save_figure_all_formats(
        fig,
        output_path,
    )

    plt.show()
    plt.close(
        fig
    )

    return output_path


# =============================================================================
# SUPPLEMENTARY FIGURE S2 — SUBGROUP SPECIFICITY AND PPV
# =============================================================================

def create_subgroup_operating_figure(
    subgroup_table: pd.DataFrame,
    audit_model: str,
    n_bootstrap: int,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    configure_style()

    plot_table = (
        subgroup_table.loc[
            subgroup_table[
                "display_order"
            ].lt(
                999
            )
        ]
        .sort_values(
            "display_order"
        )
        .reset_index(
            drop=True
        )
    )

    cmap = cm.get_cmap(
        CMAP_NAME
    )

    variable_colors = {
        variable: cmap(
            value
        )
        for variable, value in zip(
            VARIABLE_ORDER,
            np.linspace(
                0.10,
                0.90,
                len(
                    VARIABLE_ORDER
                ),
            ),
        )
    }

    def forest_metric(
        axis: plt.Axes,
        metric: str,
        title: str,
        xlabel: str,
        x_limit: Tuple[
            float,
            float,
        ],
        show_ylabels: bool,
        show_prevalence: bool = False,
    ) -> None:
        displayed = (
            plot_table.iloc[
                ::-1
            ]
            .reset_index(
                drop=True
            )
        )

        for index, row in displayed.iterrows():
            estimate = float(
                row[
                    metric
                ]
            )
            lower = float(
                row[
                    f"{metric}_ci_lower_95"
                ]
            )
            upper = float(
                row[
                    f"{metric}_ci_upper_95"
                ]
            )

            if not np.isfinite(
                estimate
            ):
                continue

            low_event = int(
                row[
                    "events"
                ]
            ) < MIN_EVENTS_FLAG

            color = variable_colors.get(
                str(
                    row[
                        "subgroup_variable"
                    ]
                ),
                cmap(
                    0.5
                ),
            )

            marker = (
                "X"
                if low_event
                else "o"
            )

            alpha = (
                0.45
                if low_event
                else 1.0
            )

            clipped_lower = (
                max(
                    lower,
                    x_limit[
                        0
                    ],
                )
                if np.isfinite(
                    lower
                )
                else estimate
            )

            clipped_upper = (
                min(
                    upper,
                    x_limit[
                        1
                    ],
                )
                if np.isfinite(
                    upper
                )
                else estimate
            )

            x_error = np.asarray(
                [
                    [
                        max(
                            0.0,
                            estimate
                            - clipped_lower,
                        )
                    ],
                    [
                        max(
                            0.0,
                            clipped_upper
                            - estimate,
                        )
                    ],
                ]
            )

            axis.errorbar(
                estimate,
                index,
                xerr=x_error,
                fmt=marker,
                markersize=6.5,
                capsize=2.5,
                linewidth=1.35,
                color=color,
                alpha=alpha,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )

            if show_prevalence:
                axis.scatter(
                    float(
                        row[
                            "prevalence"
                        ]
                    ),
                    index,
                    marker="|",
                    s=85,
                    linewidths=1.8,
                    color="dimgray",
                    alpha=0.85,
                    zorder=2,
                )

        axis.set_yticks(
            np.arange(
                len(
                    displayed
                )
            )
        )

        if show_ylabels:
            axis.set_yticklabels(
                displayed[
                    "display_label_with_counts"
                ],
                fontsize=11.0,
            )
        else:
            axis.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )

        axis.set_xlim(
            *x_limit
        )
        axis.set_xlabel(
            xlabel
        )
        axis.set_title(
            title,
            fontweight="bold",
            loc="left",
        )
        axis.grid(
            axis="x",
            linestyle="--",
            alpha=0.18,
        )

    ppv_upper_values = pd.to_numeric(
        plot_table[
            "PPV_ci_upper_95"
        ],
        errors="coerce",
    ).dropna()

    prevalence_max = float(
        plot_table[
            "prevalence"
        ].max()
    )

    ppv_upper = max(
        0.05,
        prevalence_max
        * 1.25,
        (
            float(
                ppv_upper_values.max()
            )
            * 1.08
            if len(
                ppv_upper_values
            )
            else 0.10
        ),
    )

    ppv_upper = min(
        1.0,
        ppv_upper,
    )

    fig, (
        axis_a,
        axis_b,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            20,
            10.2,
        ),
        sharey=True,
    )

    forest_metric(
        axis_a,
        "specificity",
        "A. Specificity across prespecified subgroups",
        "Specificity with bootstrap 95% CI",
        (
            0.0,
            1.02,
        ),
        True,
    )

    forest_metric(
        axis_b,
        "PPV",
        "B. Positive predictive value across prespecified subgroups",
        "Positive predictive value with bootstrap 95% CI",
        (
            0.0,
            ppv_upper,
        ),
        False,
        show_prevalence=True,
    )

    fig.suptitle(
        "Exploratory subgroup specificity and positive predictive value",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    # Audit model, threshold, and bootstrap details are reported in the caption.

    legend_handles = [
        Line2D(
            [
                0
            ],
            [
                0
            ],
            marker="o",
            color=variable_colors[
                variable
            ],
            linestyle="none",
            markersize=7,
            label=variable,
        )
        for variable in VARIABLE_ORDER
    ]

    legend_handles.extend(
        [
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                marker="X",
                color="gray",
                linestyle="none",
                markersize=7,
                label=(
                    f"Exploratory: <{MIN_EVENTS_FLAG} deaths"
                ),
            ),
            Line2D(
                [
                    0
                ],
                [
                    0
                ],
                marker="|",
                color="dimgray",
                linestyle="none",
                markersize=10,
                markeredgewidth=1.8,
                label="Subgroup prevalence in Panel B",
            ),
        ]
    )

    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
        fontsize=11.0,
    )

    # PPV prevalence dependence and interval methods are reported in the caption.

    fig.subplots_adjust(
        left=0.34,
        right=0.985,
        bottom=0.12,
        top=0.88,
        wspace=0.10,
    )

    output_path = (
        output_dir
        / "FigureS2_Subgroup_Specificity_PPV_StrictOOF.png"
    )

    save_figure_all_formats(
        fig,
        output_path,
    )

    plt.show()
    plt.close(
        fig
    )

    return output_path


# =============================================================================
# SUPPLEMENTARY FIGURE S3 — TAIL-FOCUSED CALIBRATION
# =============================================================================

def create_calibration_figure(
    calibration_bins: pd.DataFrame,
    calibration_summary: Mapping[str, Any],
    audit_model: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    configure_style()

    cmap = cm.get_cmap(
        CMAP_NAME
    )

    fig, (
        axis_a,
        axis_b,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            17.5,
            8.2,
        ),
    )

    predicted = calibration_bins[
        "mean_predicted_risk"
    ].to_numpy(
        dtype=float
    )

    observed = calibration_bins[
        "observed_mortality"
    ].to_numpy(
        dtype=float
    )

    lower = calibration_bins[
        "observed_ci_lower_95"
    ].to_numpy(
        dtype=float
    )

    upper = calibration_bins[
        "observed_ci_upper_95"
    ].to_numpy(
        dtype=float
    )

    x_error = np.vstack(
        [
            predicted
            - calibration_bins[
                "minimum_predicted_risk"
            ].to_numpy(
                dtype=float
            ),
            calibration_bins[
                "maximum_predicted_risk"
            ].to_numpy(
                dtype=float
            )
            - predicted,
        ]
    )

    y_error = np.vstack(
        [
            observed
            - lower,
            upper
            - observed,
        ]
    )

    positive_x = predicted[
        predicted > 0
    ]

    minimum_x = (
        max(
            float(
                positive_x.min()
            )
            * 0.70,
            1e-7,
        )
        if len(
            positive_x
        )
        else 1e-7
    )

    maximum_x = max(
        float(
            predicted.max()
        )
        * 1.30,
        minimum_x
        * 10.0,
    )

    ideal_x = np.logspace(
        np.log10(
            minimum_x
        ),
        np.log10(
            maximum_x
        ),
        300,
    )

    axis_a.plot(
        ideal_x,
        ideal_x,
        linestyle=":",
        linewidth=1.2,
        color="gray",
        label="Ideal calibration",
    )

    axis_a.errorbar(
        predicted,
        observed,
        xerr=x_error,
        yerr=y_error,
        fmt="o-",
        markersize=6,
        capsize=2.5,
        linewidth=1.8,
        color=cmap(
            0.55
        ),
        markerfacecolor=cmap(
            0.75
        ),
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=audit_model,
    )

    axis_a.set_xscale(
        "log"
    )
    axis_a.set_xlim(
        minimum_x,
        maximum_x,
    )
    axis_a.set_ylim(
        bottom=0.0,
    )
    axis_a.set_xlabel(
        "Mean predicted mortality probability — log scale"
    )
    axis_a.set_ylabel(
        "Observed 30-day mortality"
    )
    axis_a.set_title(
        "A. Tail-focused reliability assessment",
        fontweight="bold",
        loc="left",
    )
    axis_a.grid(
        linestyle="--",
        alpha=0.18,
    )
    axis_a.legend(
        frameon=False,
        fontsize=11.5,
    )

    annotation = (
        f"Intercept={calibration_summary['calibration_intercept']:.3f}\n"
        f"Slope={calibration_summary['calibration_slope']:.3f}\n"
        f"O/E={calibration_summary['observed_expected_ratio']:.3f}\n"
        f"Brier={calibration_summary['Brier']:.6f}"
    )

    axis_a.text(
        0.04,
        0.96,
        annotation,
        transform=axis_a.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.90,
        },
    )

    # Panel B — bin counts and deaths
    x_positions = np.arange(
        len(
            calibration_bins
        )
    )

    bars = axis_b.bar(
        x_positions,
        calibration_bins[
            "n"
        ],
        color=[
            cmap(
                value
            )
            for value in np.linspace(
                0.12,
                0.88,
                len(
                    calibration_bins
                ),
            )
        ],
        edgecolor="none",
        alpha=0.85,
        label="Encounters",
    )

    axis_b_deaths = axis_b.twinx()

    axis_b_deaths.plot(
        x_positions,
        calibration_bins[
            "events"
        ],
        marker="o",
        linewidth=1.8,
        color="black",
        label="Deaths",
    )

    axis_b.set_xticks(
        x_positions
    )
    axis_b.set_xticklabels(
        calibration_bins[
            "risk_bin"
        ].astype(
            int
        )
    )
    axis_b.set_xlabel(
        "Ascending predicted-risk quantile bin"
    )
    axis_b.set_ylabel(
        "Encounters per bin"
    )
    axis_b_deaths.set_ylabel(
        "Deaths per bin"
    )

    maximum_deaths = max(
        float(
            calibration_bins[
                "events"
            ].max()
        ),
        1.0,
    )

    axis_b_deaths.set_ylim(
        0.0,
        maximum_deaths
        * 1.16,
    )
    axis_b.set_title(
        "B. Held-out test composition across risk bins",
        fontweight="bold",
        loc="left",
        pad=16,
    )
    axis_b.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
    )

    # Label death counts next to the black markers instead of above the
    # encounter bars. This prevents the annotations from overlapping the
    # panel title while keeping the event counts directly linked to the
    # right-hand deaths axis.
    for x_position, events in zip(
        x_positions,
        calibration_bins[
            "events"
        ],
    ):
        axis_b_deaths.annotate(
            f"{int(events)}",
            xy=(
                x_position,
                float(events),
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="black",
        )

    handles_left, labels_left = (
        axis_b.get_legend_handles_labels()
    )
    handles_right, labels_right = (
        axis_b_deaths.get_legend_handles_labels()
    )

    axis_b.legend(
        handles_left
        + handles_right,
        labels_left
        + labels_right,
        frameon=False,
        fontsize=11.5,
        loc="upper left",
    )

    fig.suptitle(
        "Tail-focused calibration of the mortality model",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )


    fig.tight_layout(
        rect=(
            0.03,
            0.05,
            0.99,
            0.95,
        ),
        w_pad=2.8,
    )

    output_path = (
        output_dir
        / "FigureS3_Tail_Focused_Calibration_StrictOOF.png"
    )

    save_figure_all_formats(
        fig,
        output_path,
    )

    plt.show()
    plt.close(
        fig
    )

    return output_path


# =============================================================================
# MANUSCRIPT NUMERICAL SUMMARY
# =============================================================================

def metric_ci_text(
    row: pd.Series,
    metric: str,
    digits: int = 3,
) -> str:
    estimate = float(
        row[
            metric
        ]
    )
    lower = float(
        row[
            f"{metric}_ci_lower_95"
        ]
    )
    upper = float(
        row[
            f"{metric}_ci_upper_95"
        ]
    )

    if not (
        np.isfinite(
            estimate
        )
        and np.isfinite(
            lower
        )
        and np.isfinite(
            upper
        )
    ):
        return "not estimable"

    return (
        f"{estimate:.{digits}f} "
        f"(95% CI {lower:.{digits}f}–{upper:.{digits}f})"
    )


def build_manuscript_text(
    y_true: np.ndarray,
    threshold: float,
    audit_model: str,
    threshold_transport_mode: str,
    error_counts: pd.DataFrame,
    subgroup_table: pd.DataFrame,
    calibration_summary: Mapping[str, Any],
    false_negative_review: pd.DataFrame,
) -> str:
    lines = [
        "FINAL TEMPORAL-SAFE STRICT-OOF SUPPLEMENTARY NUMERICAL RESULTS",
        "=" * 86,
        "",
        (
            f"The held-out test cohort included {len(y_true):,} encounters "
            f"and {int(y_true.sum())} deaths "
            f"({100.0 * float(y_true.mean()):.3f}%)."
        ),
        (
            f"The {audit_model} was evaluated at the "
            f"development-defined threshold of {threshold:.8f} using "
            f"{threshold_transport_mode} threshold transport."
        ),
        "",
        "ERROR ANALYSIS",
    ]

    for row in error_counts.itertuples():
        lines.append(
            f"{row.error_group}: {int(row.n):,}."
        )

    lines.append(
        (
            f"Deidentified false-negative case review rows: "
            f"{len(false_negative_review)}."
        )
    )

    lines.extend(
        [
            "",
            "TAIL-FOCUSED CALIBRATION",
            (
                "Calibration was summarized using equal-frequency predicted-risk "
                "quantile bins with Wilson 95% confidence intervals for observed "
                "mortality."
            ),
            (
                f"Calibration intercept: "
                f"{calibration_summary['calibration_intercept']:.3f}."
            ),
            (
                f"Calibration slope: "
                f"{calibration_summary['calibration_slope']:.3f}."
            ),
            (
                f"Observed/expected mortality ratio: "
                f"{calibration_summary['observed_expected_ratio']:.3f}."
            ),
            (
                f"Brier score: "
                f"{calibration_summary['Brier']:.6f}."
            ),
            "",
            "SUBGROUP SPECIFICITY AND PPV",
        ]
    )

    main_subgroups = (
        subgroup_table.loc[
            subgroup_table[
                "display_order"
            ].lt(
                999
            )
        ]        .sort_values(
            "display_order"
        )
    )

    for _, row in main_subgroups.iterrows():
        lines.append(
            (
                f"{row['display_label']}: "
                f"N={int(row['n']):,}; deaths={int(row['events'])}; "
                f"specificity {metric_ci_text(row, 'specificity')}; "
                f"PPV {metric_ci_text(row, 'PPV')}; "
                f"prevalence={float(row['prevalence']):.4f}."
            )
        )

    lines.extend(
        [
            "",
            (
                f"All subgroup estimates are exploratory because the holdout "
                f"cohort contains only {int(y_true.sum())} deaths."
            ),
        ]
    )

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_all(
    run_dir: pathlib.Path,
    pipeline_module_path: pathlib.Path,
    data_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    audit_model: str,
    n_bootstrap: int,
    n_jobs: int,
    n_calibration_bins: int,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF SUPPLEMENTARY ANALYSES")
    print("=" * 100)
    print("Run directory   :", run_dir)
    print("Pipeline module :", pipeline_module_path)
    print("Output directory:", output_dir)

    pipeline = load_pipeline_module(
        pipeline_module_path
    )

    reconstructed = reconstruct_holdout(
        pipeline,
        run_dir,
        data_path,
    )

    # This supplementary audit focuses on one reference model at the saved
    # development-defined operating point. Hybrid DARN-XGB Blend is the default
    # for continuity with the manuscript operating-point analyses; this is not
    # a claim of model superiority.
    resolved_audit_model = (
        DEFAULT_AUDIT_MODEL
        if str(audit_model).strip().lower() in {"", "auto"}
        else str(audit_model)
    )

    audit = load_audit_model(
        run_dir,
        reconstructed[
            "predictions"
        ],
        resolved_audit_model,
    )
    resolved_audit_model = str(audit["model_row"]["model"])
    threshold_transport_mode = str(
        reconstructed["config"].get(
            "threshold_transport_mode",
            "absolute",
        )
    )

    y_true = reconstructed[
        "y_true"
    ]

    probability = audit[
        "probability"
    ]

    threshold = float(
        audit[
            "threshold"
        ]
    )

    holdout = reconstructed[
        "holdout"
    ]

    subgroups = derive_subgroups(
        holdout
    )

    print("\n" + "=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF REVIEWER SUPPLEMENTARY ANALYSES")
    print("=" * 100)
    print("Run directory:", run_dir)
    print("Audit model:", resolved_audit_model)
    print(
        "Probability column:",
        audit[
            "probability_column"
        ],
    )
    print(
        f"Threshold: {threshold:.8f} ({THRESHOLD_LABEL})"
    )
    print(
        f"Held-out test: N={len(y_true):,}; deaths={int(y_true.sum())}"
    )

    # -------------------------------------------------------------------------
    # Error analysis
    # -------------------------------------------------------------------------
    error_groups = assign_error_groups(
        y_true,
        probability,
        threshold,
    )

    error_counts = (
        pd.Series(
            error_groups
        )
        .value_counts()
        .reindex(
            ERROR_GROUP_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "error_group"
        )
        .reset_index(
            name="n"
        )
    )

    error_profile = build_error_profile(
        holdout,
        error_groups,
        probability,
        subgroups,
    )

    false_negative_review = (
        build_false_negative_review(
            holdout,
            y_true,
            probability,
            threshold,
            error_groups,
            subgroups,
        )
    )

    # Deidentified audit table. Raw source rows and direct identifiers are not
    # exported.
    all_error_cases = pd.DataFrame(
        {
            "holdout_row_position": np.arange(len(holdout), dtype=int),
            "observed_outcome": y_true.astype(int),
            "error_group": error_groups,
            "predicted_probability": probability,
            "threshold": np.full(len(holdout), threshold, dtype=float),
            "distance_from_threshold": probability - threshold,
            "sex_group": (
                subgroups["Sex"].astype(str).to_numpy()
                if "Sex" in subgroups
                else np.full(len(holdout), "Unavailable", dtype=object)
            ),
            "age_group": (
                subgroups["Age"].astype(str).to_numpy()
                if "Age" in subgroups
                else np.full(len(holdout), "Unavailable", dtype=object)
            ),
            "asa_group": (
                subgroups["ASA"].astype(str).to_numpy()
                if "ASA" in subgroups
                else np.full(len(holdout), "Unavailable", dtype=object)
            ),
            "urgency_group": (
                subgroups["Urgency"].astype(str).to_numpy()
                if "Urgency" in subgroups
                else np.full(len(holdout), "Unavailable", dtype=object)
            ),
            "raw_missing_count": holdout.isna().sum(axis=1).to_numpy(dtype=int),
            "raw_missing_fraction": holdout.isna().mean(axis=1).to_numpy(dtype=float),
        }
    )


    available_capture_probabilities = (
        resolve_available_model_probabilities(
            reconstructed[
                "predictions"
            ],
            CAPTURE_MODEL_ORDER,
        )
    )

    if resolved_audit_model not in available_capture_probabilities:
        available_capture_probabilities[
            resolved_audit_model
        ] = probability

    capture_curve, capture_summary = (
        build_capture_curves(
            y_true,
            available_capture_probabilities,
            threshold,
            resolved_audit_model,
        )
    )

    error_figure = create_error_figure(
        y_true,
        probability,
        threshold,
        error_groups,
        error_profile,
        capture_curve,
        capture_summary,
        resolved_audit_model,
        output_dir,
    )

    error_counts_path = (
        output_dir
        / "FigS_Error_Group_Counts_StrictOOF.csv"
    )
    error_profile_path = (
        output_dir
        / "FigS_Error_Group_Clinical_Profile_StrictOOF.csv"
    )
    false_negative_path = (
        output_dir
        / "TableS_Deidentified_False_Negative_Review_StrictOOF.csv"
    )
    all_error_cases_path = (
        output_dir
        / "TableS_Deidentified_Error_Case_Audit_StrictOOF.csv"
    )
    capture_curve_path = (
        output_dir
        / "FigS_Death_Capture_Alert_Burden_Curves_StrictOOF.csv"
    )
    capture_summary_path = (
        output_dir
        / "TableS_Death_Capture_Alert_Burden_StrictOOF.csv"
    )

    error_counts.to_csv(
        error_counts_path,
        index=False,
    )
    error_profile.to_csv(
        error_profile_path,
        index=False,
    )
    false_negative_review.to_csv(
        false_negative_path,
        index=False,
    )
    all_error_cases.to_csv(
        all_error_cases_path,
        index=False,
    )
    capture_curve.to_csv(
        capture_curve_path,
        index=False,
    )
    capture_summary.to_csv(
        capture_summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Subgroup specificity and PPV
    # -------------------------------------------------------------------------
    subgroup_table = build_subgroup_operating_table(
        y_true,
        probability,
        threshold,
        subgroups,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
    )

    subgroup_figure = (
        create_subgroup_operating_figure(
            subgroup_table,
            resolved_audit_model,
            n_bootstrap,
            output_dir,
        )
    )

    subgroup_table_path = (
        output_dir
        / "FigureS2_Subgroup_Specificity_PPV_Bootstrap_StrictOOF.csv"
    )

    subgroup_table.to_csv(
        subgroup_table_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Tail-focused calibration
    # -------------------------------------------------------------------------
    calibration_bins = (
        build_tail_calibration_table(
            y_true,
            probability,
            n_calibration_bins,
        )
    )

    calibration_fit = (
        fit_calibration_intercept_slope(
            y_true,
            probability,
        )
    )

    calibration_summary: Dict[
        str,
        Any,
    ] = {
        "audit_model": resolved_audit_model,
        "threshold": threshold,
        "threshold_transport_mode": threshold_transport_mode,
        **calibration_fit,
        "observed_events": int(
            y_true.sum()
        ),
        "expected_events": float(
            probability.sum()
        ),
        "observed_expected_ratio": safe_divide(
            float(
                y_true.sum()
            ),
            float(
                probability.sum()
            ),
        ),
        "mean_predicted_risk": float(
            probability.mean()
        ),
        "observed_prevalence": float(
            y_true.mean()
        ),
        "Brier": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
        "calibration_bin_method": (
            "Equal-frequency quantile bins based on ranked predicted "
            "probabilities"
        ),
        "calibration_interval_method": (
            "Wilson score 95% confidence intervals for observed mortality"
        ),
        "calibration_horizontal_interval": (
            "Minimum-to-maximum predicted probability within each bin"
        ),
        "AUROC": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "AUPRC": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
    }

    calibration_figure = (
        create_calibration_figure(
            calibration_bins,
            calibration_summary,
            resolved_audit_model,
            output_dir,
        )
    )

    calibration_bins_path = (
        output_dir
        / "FigureS3_Tail_Calibration_Quantile_Bins_StrictOOF.csv"
    )
    calibration_summary_path = (
        output_dir
        / "FigureS3_Calibration_Parameters_StrictOOF.json"
    )

    calibration_bins.to_csv(
        calibration_bins_path,
        index=False,
    )

    save_json(
        calibration_summary,
        calibration_summary_path,
    )

    # -------------------------------------------------------------------------
    # Manuscript text and master summary
    # -------------------------------------------------------------------------
    manuscript_text = build_manuscript_text(
        y_true,
        threshold,
        resolved_audit_model,
        threshold_transport_mode,
        error_counts,
        subgroup_table,
        calibration_summary,
        false_negative_review,
    )

    manuscript_path = (
        output_dir
        / "Reviewer_Supplementary_Manuscript_Numerics_StrictOOF.txt"
    )

    manuscript_path.write_text(
        manuscript_text,
        encoding="utf-8",
    )

    master_summary = {
        "analysis_version": "temporal-safe-strict-oof",
        "run_id": run_dir.name,
        "run_directory": str(
            run_dir
        ),
        "pipeline_module": str(
            pipeline_module_path
        ),
        "data_path": str(
            reconstructed[
                "data_path"
            ]
        ),
        "audit_model": resolved_audit_model,
        "probability_column": audit[
            "probability_column"
        ],
        "threshold": threshold,
        "threshold_label": THRESHOLD_LABEL,
        "threshold_transport_mode": threshold_transport_mode,
        "threshold_source": (
            "Development out-of-fold predictions; threshold read from "
            "test_model_metrics.csv"
        ),
        "test_n": int(
            len(
                y_true
            )
        ),
        "test_events": int(
            y_true.sum()
        ),
        "test_prevalence": float(
            y_true.mean()
        ),
        "bootstrap_replicates_per_subgroup": int(
            n_bootstrap
        ),
        "calibration_bins": int(
            len(
                calibration_bins
            )
        ),
        "error_figure": str(
            error_figure
        ),
        "subgroup_figure": str(
            subgroup_figure
        ),
        "calibration_figure": str(
            calibration_figure
        ),
        "false_negative_count": int(
            (
                error_groups
                == "False negative"
            ).sum()
        ),
        "decision_curve_repeated": False,
        "error_case_export": (
            "Deidentified audit fields only; raw holdout rows and direct "
            "identifiers are not exported."
        ),
        "supplementary_figure_S2": {
            "figure": str(subgroup_figure),
            "analysis": "Subgroup specificity and PPV",
            "interval_method": (
                "Patient-level stratified bootstrap within each subgroup"
            ),
            "bootstrap_replicates_per_subgroup": int(n_bootstrap),
            "PPV_prevalence_note": (
                "PPV is prevalence-dependent; subgroup prevalence is displayed "
                "in Panel B"
            ),
        },
        "supplementary_figure_S3": {
            "figure": str(calibration_figure),
            "analysis": "Tail-focused calibration",
            "binning_method": (
                "Equal-frequency quantile bins based on ranked predicted "
                "probabilities"
            ),
            "observed_interval_method": (
                "Wilson score 95% confidence intervals"
            ),
            "horizontal_interval_definition": (
                "Minimum-to-maximum predicted probability within bin"
            ),
        },
        "bootstrap_method": (
            "Patient-level stratified bootstrap within each subgroup."
        ),
        "interpretation_note": (
            "These analyses support internal auditing of the locked "
            "strict-OOF held-out test cohort. Small event counts and wide "
            "intervals preclude claims of clinical safety, demographic "
            "fairness, or external validity."
        ),
    }

    master_summary_path = (
        output_dir
        / "Reviewer_Supplementary_StrictOOF_Summary.json"
    )

    save_json(
        master_summary,
        master_summary_path,
    )

    print("\n" + "=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF REVIEWER SUPPLEMENTARY ANALYSES COMPLETE")
    print("=" * 100)

    print("\nFigures:")
    print(error_figure)
    print(subgroup_figure)
    print(calibration_figure)

    print("\nKey tables:")
    print(false_negative_path)
    print(capture_summary_path)
    print(subgroup_table_path)
    print(calibration_bins_path)
    print(calibration_summary_path)

    print("\nSummary:")
    print(master_summary_path)

    print("\n" + "=" * 100)
    print("MANUSCRIPT-READY NUMERICS")
    print("=" * 100)
    print(manuscript_text)


# =============================================================================
# JUPYTER-SAFE ARGUMENT PARSING
# =============================================================================

def parse_args(
    argv: Optional[
        Sequence[str]
    ] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Generate final temporal-safe strict-OOF supplementary analyses, including "
            "numbered Figure S2 subgroup specificity/PPV and Figure S3 "
            "quantile-bin tail calibration."
        ),
    )

    parser.add_argument(
        "--run_dir",
        type=pathlib.Path,
        default=DEFAULT_RUN_DIR,
    )

    parser.add_argument(
        "--pipeline_module",
        type=pathlib.Path,
        default=DEFAULT_PIPELINE_MODULE,
    )

    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
    )

    parser.add_argument(
        "--audit_model",
        "--primary_model",
        dest="audit_model",
        type=str,
        default="auto",
        help=(
            "Model used for the supplementary audit. Default: Hybrid DARN-XGB Blend. "
            "The legacy --primary_model flag is accepted as an alias."
        ),
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=N_BOOTSTRAP,
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=N_JOBS,
    )

    parser.add_argument(
        "--n_calibration_bins",
        type=int,
        default=N_CALIBRATION_BINS,
    )

    raw_arguments = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    filtered_arguments: List[
        str
    ] = []

    skip_next = False

    for argument in raw_arguments:
        if skip_next:
            skip_next = False
            continue

        if argument in {
            "-f",
            "--f",
        }:
            skip_next = True
            continue

        if (
            argument.startswith(
                "-f="
            )
            or argument.startswith(
                "--f="
            )
        ):
            continue

        filtered_arguments.append(
            argument
        )

    args, unknown = parser.parse_known_args(
        filtered_arguments
    )

    if unknown:
        print(
            "Ignoring unrecognized Jupyter/IPython arguments:",
            " ".join(
                map(
                    str,
                    unknown,
                )
            ),
        )

    return args


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            args.run_dir
            / DEFAULT_OUTPUT_SUBDIR
        )
    )

    run_all(
        run_dir=args.run_dir,
        pipeline_module_path=args.pipeline_module,
        data_path=args.data_path,
        output_dir=output_dir,
        audit_model=args.audit_model,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        n_calibration_bins=args.n_calibration_bins,
    )


if __name__ == "__main__":
    main()