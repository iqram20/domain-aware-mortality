#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build all manuscript tables from the FINAL temporal-safe + strict-OOF run,
without hard-coding numerical results, and compile current-run figures into
a Microsoft Word document.

Default final run
-----------------
/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/
darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653

What this script creates
------------------------
Main tables
  Table 1   Cohort characteristics
  Table 2   Held-out test model performance

Supplementary tables
  Table S1a Cohort flow
  Table S1b Development/test split audit
  Table S2a Fold-level feature-selection summary
  Table S2b Selected features by clinical domain and fold
  Table S3  Full saved held-out test model metrics
  Table S4  Saved paired-bootstrap model comparisons
  Table S5  Uniform DARN fitted-model input-domain masking
  Table S6a Leave-one-domain-out retrained test metrics
  Table S6b Leave-one-domain-out paired-bootstrap differences
  Table S7  Exploratory subgroup performance / operating-point audit
  Table S8  Tail-focused calibration bins
  Table S9  TP/FP/FN/TN descriptive clinical profiles
  Table S10 Deidentified false-negative review
  Table S11 Component-level explainability features (DARN + XGBoost)
  Table S12a Eye/Ear adjusted association, if present
  Table S12b Eye/Ear feature-removal test metrics, if present
  Table S12c Eye/Ear feature-removal paired-bootstrap comparison, if present

Compilation files
  All_Tables_TemporalSafe_StrictOOF.docx
  Submission_Figures_TemporalSafe_StrictOOF.docx
      Exact 12-figure package: Figure 1–6 + Figure S1–S6
  All_Figures_Archive_TemporalSafe_StrictOOF.docx
      Internal archive only; not for journal submission
  Publication_Table_Manifest.csv
  Figure_Submission_Manifest.csv

Row-level probability/prediction files are retained in the analysis run but
are intentionally excluded from the manuscript supplementary-table package.

Publication display rounding
----------------------------
Raw CSV outputs remain full precision. Word tables use:
- counts: integers with commas
- descriptive continuous values: 1 decimal
- descriptive percentages: 1 decimal
- mortality prevalence: 2 decimals (%)
- AUROC/AUPRC and 95% CIs: 3 decimals
- paired ΔAUROC/ΔAUPRC and CIs: 4 decimals
- Brier: 5 decimals
- paired ΔBrier and CIs: 6 decimals
- sensitivity/specificity/PPV/NPV/alert rate: 1 decimal (%)
- thresholds: 5 decimals
- calibration slope/intercept/O:E: 3 decimals
- ECE: 5 decimals
- OR/HR and CIs: 2 decimals
- p values: 3 decimals, with <0.001 when smaller

Compact Word presentation
-------------------------
The Word compilation is intentionally narrower than the CSV artifacts:
- Table S3 is split into discrimination/calibration and operating-point sections.
- Tables S4, S6b, and S12c combine paired differences and 95% CIs.
- Table S6a is reduced to domain/model/AUROC/AUPRC/Brier.
- Table S7 is split into subgroup discrimination and operating-point sections.
- Table S8 combines observed mortality and its 95% CI.
- Table S11 is split by model component.
- Tables S12a-S12c use compact publication-facing layouts.
No analysis columns are deleted from the CSV files.

Publication-table completeness controls
---------------------------------------
The final Word build merges bootstrap_confidence_intervals.csv into the held-out
model metrics so AUROC/AUPRC confidence intervals are actually shown. Recall is
normalized to Sensitivity, alert burden is reconstructed from TP+FP when needed,
S5/S6b/S12c use dedicated compact views, S12a excludes descriptive rows without
odds ratios and retains the model-stage label, and S10 renders genuine source
missingness as NA. Unexpected blank Word cells in other tables stop the build.

Table S4 source isolation
-------------------------
Table S4 is constructed only from the canonical root-level overall model
paired-bootstrap artifacts. Eye/Ear feature-removal analyses, leave-one-domain-
out retraining, figure-derived duplicates, and publication-package outputs are
excluded. The Word display is fixed to Comparison, Metric, Difference (95% CI),
p-value, and Bootstrap replicates.

Fixed table sequence
--------------------
The manifest and Word table document are always ordered as:
Table 1, Table 2, Table S1a, Table S1b, Table S2a, Table S2b, Table S3,
Table S4, Table S5, Table S6a, Table S6b, Table S7, Table S8, Table S9,
Table S10, Table S11, Table S12a, Table S12b, Table S12c.
This ordering is independent of the order in which tables are generated.

Word-table safety
-----------------
If a compact Word layout resolves too few columns or produces an effectively
empty data view, the script automatically falls back to the complete formatted
table and splits it horizontally into blocks of at most 8 columns. Identifier
columns are repeated across parts. Paired-comparison tables (S4, S6b, S12c)
must contain actual statistical result columns; a label-only view such as
Model + Metric is rejected and replaced by the complete paired-bootstrap
results. Decimal control is applied semantically to prefixed/suffixed metric
columns and to combined confidence-interval strings. This prevents raw
machine-precision values from leaking into the Word document while preserving
full precision in the CSV artifacts.

Important
---------
- Numerical results are read from the run artifacts or computed from the
  saved predictions and temporal-safe cohort. They are not manually entered.
- The Hybrid DARN-XGB Blend is only the default reference model for the
  operating-point/subgroup/error/calibration audit. This is not a claim that
  it is a superior or "primary" model. Use --audit_model to change it.
- Existing saved paired-bootstrap, masking, LODO, and explainability artifacts
  are preferred so the tables exactly reflect the completed run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# DEFAULT CURRENT FINAL RUN
# =============================================================================

DEFAULT_PROJECT_ROOT = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire"
)

DEFAULT_RUN_DIR = (
    DEFAULT_PROJECT_ROOT
    / "Results"
    / "darn_cv_baseline_runs"
    / "v6_temporal_safe_strict_oof_parallel"
    / "run_20260919_115653"
)

DEFAULT_PIPELINE_MODULE = (
    DEFAULT_PROJECT_ROOT
    / "Code"
    / "New"
    / "Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py"
)

DEFAULT_DATA_PATH = (
    DEFAULT_PROJECT_ROOT
    / "Results"
    / "Final_Datasets"
    / "df_final_mortality_2026-05-24_12-44_temporal_safe.csv"
)

DEFAULT_AUDIT_MODEL = "Hybrid DARN-XGB Blend"

DEFAULT_OUTPUT_SUBDIR = "publication_tables_temporal_safe_strict_oof"

DEFAULT_BOOTSTRAPS = 2000
DEFAULT_JOBS = 8
DEFAULT_RANDOM_STATE = 20260919
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_TOP_EXPLAINABILITY_FEATURES = 25


# =============================================================================
# FIXED MANUSCRIPT TABLE ORDER
# =============================================================================
#
# This order is used for BOTH:
#   1) Publication_Table_Manifest.csv
#   2) All_Tables_TemporalSafe_StrictOOF.docx
#
# It is independent of the order in which tables are generated.
TABLE_DOCUMENT_ORDER = [
    "Table 1",
    "Table 2",
    "Table S1a",
    "Table S1b",
    "Table S2a",
    "Table S2b",
    "Table S3",
    "Table S4",
    "Table S5",
    "Table S6a",
    "Table S6b",
    "Table S7",
    "Table S8",
    "Table S9",
    "Table S10",
    "Table S11",
    "Table S12a",
    "Table S12b",
    "Table S12c",
]

TABLE_DOCUMENT_ORDER_MAP = {
    table_id: index
    for index, table_id in enumerate(TABLE_DOCUMENT_ORDER)
}


def sort_publication_manifest(
    manifest_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the publication manifest in fixed manuscript order.

    Known table IDs follow TABLE_DOCUMENT_ORDER exactly. Any unexpected/new
    table IDs are retained and appended afterward in their original generation
    order, so no table is silently lost.
    """
    if manifest_frame is None or manifest_frame.empty:
        return manifest_frame

    result = manifest_frame.copy()
    result["_generation_order"] = np.arange(len(result))
    result["_document_order"] = (
        result["table_id"]
        .map(TABLE_DOCUMENT_ORDER_MAP)
        .fillna(len(TABLE_DOCUMENT_ORDER))
        .astype(int)
    )

    result = (
        result.sort_values(
            ["_document_order", "_generation_order"],
            kind="stable",
        )
        .drop(
            columns=[
                "_document_order",
                "_generation_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result



# =============================================================================
# EXPLICIT MANUSCRIPT FIGURE SET
# =============================================================================
#
# Figures 3–6 and S1–S6 are resolved by exact basename from the final run.
# Figure 1 and Figure 2 are not present in the final run directory and must
# either be supplied with --figure1_path / --figure2_path or be copied into
# the run directory using these canonical basenames.
#
# This explicit map intentionally replaces heuristic figure selection.

EXPLICIT_RUN_FIGURES: Dict[str, str] = {
    "Figure 3": "Figure3_TemporalSafe_StrictOOF_Performance_Clinical_Utility.png",
    "Figure 4": "Figure4_DARN_XGBoost_Component_Explainability_AB.png",
    "Figure 5": "Figure5_Domain_Attribution_Masking_and_LODO_StrictOOF.png",
    "Figure 6": "Figure6_TemporalSafe_StrictOOF_Subgroup_Performance_2x2.png",
    "Figure S1": "FigS_Error_Analysis_StrictOOF.png",
    "Figure S2": "FigureS2_Subgroup_Specificity_PPV_StrictOOF.png",
    "Figure S3": "FigureS3_Tail_Focused_Calibration_StrictOOF.png",
    "Figure S4": "FigS_TemporalSafe_StrictOOF_XAI_Input_Correlation_2x2.png",
    "Figure S5": "FigS_Leave_One_Domain_Out_Retraining.png",
    "Figure S6": "Supplementary_Figure6_Urgency_PR_Curves.png",
}

SUBMISSION_FIGURE_ORDER = [
    "Figure 1",
    "Figure 2",
    "Figure 3",
    "Figure 4",
    "Figure 5",
    "Figure 6",
    "Figure S1",
    "Figure S2",
    "Figure S3",
    "Figure S4",
    "Figure S5",
    "Figure S6",
]


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

ERROR_GROUP_ORDER = [
    "True positive",
    "False positive",
    "False negative",
    "True negative",
]

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
    "XGBoost": ["xgboost"],
    "Uniform DARN": ["uniformdarn", "darnuniformdomains", "uniformdomains"],
    "MLP": ["standardmlp", "mlp"],
    "Logistic Regression": ["logisticregression", "clinicallogistic", "fulllogistic"],
    "ASA-only Logistic": ["asaonlylogistic", "asalogistic"],
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with open(require_file(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    normalized = {normalize_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return str(candidate)
        match = normalized.get(normalize_name(candidate))
        if match is not None:
            return match
    return None


def first_existing_file(
    candidates: Iterable[pathlib.Path],
) -> Optional[pathlib.Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def load_pipeline_module(path: pathlib.Path):
    require_file(path)
    spec = importlib.util.spec_from_file_location(
        "mortality_darn_table_builder_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline module:\n{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_divide(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def save_table(
    frame: pd.DataFrame,
    path: pathlib.Path,
    *,
    index: bool = False,
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def add_manifest_entry(
    manifest: List[Dict[str, Any]],
    table_id: str,
    title: str,
    path: Optional[pathlib.Path],
    source: str,
    status: str = "created",
) -> None:
    manifest.append(
        {
            "table_id": table_id,
            "title": title,
            "file": "" if path is None else str(path),
            "source": source,
            "status": status,
        }
    )


def is_generated_publication_artifact(path: pathlib.Path) -> bool:
    """Return True for files inside this script's generated publication package."""
    return DEFAULT_OUTPUT_SUBDIR in path.parts


def discover_csvs(
    root: pathlib.Path,
    include_terms: Sequence[str],
    exclude_terms: Sequence[str] = (),
) -> List[pathlib.Path]:
    results: List[pathlib.Path] = []
    if not root.exists():
        return results

    include_norm = [term.lower() for term in include_terms]
    exclude_norm = [term.lower() for term in exclude_terms]

    for path in root.rglob("*.csv"):
        # Do not read tables produced by a previous execution of this script
        # back in as source analysis artifacts.
        if is_generated_publication_artifact(path):
            continue

        name = str(path).lower()
        if all(term in name for term in include_norm) and not any(
            term in name for term in exclude_norm
        ):
            results.append(path)

    return sorted(set(results))


# =============================================================================
# COHORT + SPLIT RECONSTRUCTION
# =============================================================================

def prediction_position_column(frame: pd.DataFrame) -> Optional[str]:
    return first_existing_column(
        frame,
        [
            "final_cohort_row_position",
            "original_row_position",
            "source_row_position",
        ],
    )


def reconstruct_current_run(
    run_dir: pathlib.Path,
    pipeline_module_path: pathlib.Path,
    data_path: pathlib.Path,
) -> Dict[str, Any]:
    pipeline = load_pipeline_module(pipeline_module_path)
    config = load_json(run_dir / "config.json")

    source = pd.read_csv(require_file(data_path), low_memory=False)
    target_column = str(config.get("target_column", "mortality_30d"))

    cohort, cohort_flow = pipeline.apply_reviewer_cohort_exclusions(
        source,
        target_column=target_column,
        exclude_asa6=bool(config.get("exclude_asa6", True)),
    )
    cohort = cohort.reset_index(drop=True)

    test_predictions = pd.read_csv(
        require_file(run_dir / "test_predictions_all_models.csv"),
        low_memory=False,
    )
    if "y_true" not in test_predictions.columns:
        raise KeyError("test_predictions_all_models.csv is missing y_true.")

    test_position_col = prediction_position_column(test_predictions)
    if test_position_col is None:
        raise KeyError(
            "Test predictions need final_cohort_row_position, "
            "original_row_position, or source_row_position."
        )

    if normalize_name(test_position_col) == normalize_name("source_row_position"):
        source_position = pd.to_numeric(
            test_predictions[test_position_col],
            errors="raise",
        ).astype(int)
        source_indexed = (
            source.reset_index(drop=False)
            .rename(columns={"index": "source_row_position"})
        )
        test_cohort = (
            source_indexed.set_index("source_row_position")
            .loc[source_position.to_numpy()]
            .reset_index(drop=True)
        )
        # For full-cohort split auditing, source positions are not equivalent
        # to corrected-cohort positions.
        test_positions_for_split = None
    else:
        test_positions_for_split = pd.to_numeric(
            test_predictions[test_position_col],
            errors="raise",
        ).astype(int).to_numpy()
        test_cohort = (
            cohort.iloc[test_positions_for_split]
            .reset_index(drop=True)
        )

    y_test = pd.to_numeric(
        test_predictions["y_true"],
        errors="raise",
    ).astype(int).to_numpy()

    y_reconstructed = pd.to_numeric(
        test_cohort[target_column],
        errors="raise",
    ).astype(int).to_numpy()

    if not np.array_equal(y_test, y_reconstructed):
        raise RuntimeError(
            "Reconstructed held-out test outcomes do not match saved predictions."
        )

    development_file = first_existing_file(
        [
            run_dir / "development_oof_predictions.csv",
            run_dir / "development_predictions.csv",
        ]
    )

    development_predictions = None
    development_positions = None

    if development_file is not None:
        development_predictions = pd.read_csv(development_file, low_memory=False)
        dev_position_col = prediction_position_column(development_predictions)
        if (
            dev_position_col is not None
            and normalize_name(dev_position_col)
            != normalize_name("source_row_position")
        ):
            development_positions = pd.to_numeric(
                development_predictions[dev_position_col],
                errors="raise",
            ).astype(int).to_numpy()

    target = pd.to_numeric(
        cohort[target_column],
        errors="raise",
    ).astype(int)

    split_rows: List[Dict[str, Any]] = []

    if development_positions is not None:
        y_dev = target.iloc[development_positions].to_numpy()
        split_rows.append(
            {
                "Cohort": "Development",
                "Encounters": int(len(development_positions)),
                "Deaths": int(y_dev.sum()),
                "Mortality_prevalence_percent": (
                    100.0 * float(y_dev.mean())
                ),
            }
        )

    split_rows.append(
        {
            "Cohort": "Held-out test",
            "Encounters": int(len(y_test)),
            "Deaths": int(y_test.sum()),
            "Mortality_prevalence_percent": 100.0 * float(y_test.mean()),
        }
    )

    split_rows.append(
        {
            "Cohort": "Overall modeled cohort",
            "Encounters": int(len(cohort)),
            "Deaths": int(target.sum()),
            "Mortality_prevalence_percent": 100.0 * float(target.mean()),
        }
    )

    # Verify exhaustive development/test partition when corrected-cohort row
    # positions are available for both.
    if development_positions is not None and test_positions_for_split is not None:
        overlap = np.intersect1d(
            development_positions,
            test_positions_for_split,
        )
        if len(overlap):
            raise RuntimeError(
                "Development and held-out test row positions overlap."
            )

        represented = np.concatenate(
            [development_positions, test_positions_for_split]
        )
        expected = np.arange(len(cohort), dtype=int)

        if (
            len(np.unique(represented)) != len(represented)
            or len(np.setdiff1d(expected, represented)) != 0
            or len(np.setdiff1d(represented, expected)) != 0
        ):
            raise RuntimeError(
                "Development and held-out test positions do not exhaustively "
                "reconstruct the modeled cohort."
            )

    return {
        "pipeline": pipeline,
        "config": config,
        "source": source,
        "cohort": cohort,
        "cohort_flow": cohort_flow.copy(),
        "split_audit": pd.DataFrame(split_rows),
        "target_column": target_column,
        "target": target,
        "test_predictions": test_predictions,
        "test_cohort": test_cohort,
        "y_test": y_test,
        "development_predictions": development_predictions,
        "development_positions": development_positions,
        "data_path": data_path,
    }


# =============================================================================
# TABLE 1 — COHORT CHARACTERISTICS
# =============================================================================

def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def continuous_summary(series: pd.Series) -> str:
    values = numeric(series).dropna()
    if values.empty:
        return "NA"
    median = float(values.median())
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return f"{median:.1f} [{q1:.1f}–{q3:.1f}]"


def binary_summary(
    condition: pd.Series,
    valid: Optional[pd.Series] = None,
) -> str:
    condition = pd.Series(
        condition,
        index=condition.index,
    ).fillna(False).astype(bool)

    if valid is None:
        valid = pd.Series(True, index=condition.index, dtype=bool)
    else:
        valid = pd.Series(
            valid,
            index=condition.index,
        ).fillna(False).astype(bool)

    denominator = int(valid.sum())
    if denominator == 0:
        return "NA"

    count = int((condition & valid).sum())
    percentage = 100.0 * count / denominator
    return f"{count:,} ({percentage:.1f}%)"


def normalize_sex(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    numeric_values = pd.to_numeric(series, errors="coerce")

    output = pd.Series(
        "Missing/Other",
        index=series.index,
        dtype=object,
    )
    output.loc[
        numeric_values.eq(0)
        | raw.isin(["f", "female", "woman", "women", "false"])
    ] = "Female"
    output.loc[
        numeric_values.eq(1)
        | raw.isin(["m", "male", "man", "men", "true"])
    ] = "Male"
    return output


def parse_asa(series: pd.Series) -> pd.Series:
    numeric_asa = pd.to_numeric(series, errors="coerce")
    missing = numeric_asa.isna()
    if missing.any():
        extracted = (
            series.astype("string")
            .str.extract(r"(?i)(?:ASA\s*)?([1-6])", expand=False)
        )
        numeric_asa.loc[missing] = pd.to_numeric(
            extracted.loc[missing],
            errors="coerce",
        )
    return numeric_asa.astype(float)


def normalize_emergency(
    series: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    raw = series.astype(str).str.strip().str.lower()
    values = pd.to_numeric(series, errors="coerce")

    positive_text = {
        "true", "yes", "y", "emergency", "emergent", "urgent"
    }
    negative_text = {
        "false", "no", "n", "elective", "scheduled"
    }

    flag = values.eq(1) | raw.isin(positive_text)
    valid = values.isin([0, 1]) | raw.isin(
        positive_text | negative_text
    )
    return flag, valid


def derive_icu_status(
    frame: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    binary_column = first_existing_column(
        frame,
        [
            "icu_admission",
            "postop_icu_outcome",
            "postop_icu",
            "icu",
        ],
    )
    if binary_column is not None:
        values = pd.to_numeric(frame[binary_column], errors="coerce")
        return values.eq(1), values.notna(), binary_column

    icu_days_column = first_existing_column(
        frame,
        ["icu_days", "postop_icu_days"],
    )
    if icu_days_column is not None:
        values = pd.to_numeric(frame[icu_days_column], errors="coerce")
        return values.gt(0), values.notna(), icu_days_column

    icu_time_column = first_existing_column(
        frame,
        ["icuin_time", "icu_in_time"],
    )
    if icu_time_column is not None:
        valid = pd.Series(True, index=frame.index, dtype=bool)
        return frame[icu_time_column].notna(), valid, icu_time_column

    return None, None, None


def add_continuous_row(
    rows: List[List[str]],
    cohort: pd.DataFrame,
    survivors: pd.DataFrame,
    deaths: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
) -> None:
    column = first_existing_column(cohort, candidates)
    if column is None:
        return

    rows.append(
        [
            label,
            continuous_summary(cohort[column]),
            continuous_summary(survivors[column]),
            continuous_summary(deaths[column]),
        ]
    )


def build_table1(reconstructed: Mapping[str, Any]) -> pd.DataFrame:
    cohort = reconstructed["cohort"].copy()
    target_column = reconstructed["target_column"]
    cohort[target_column] = pd.to_numeric(
        cohort[target_column],
        errors="raise",
    ).astype(int)

    survivors = cohort.loc[cohort[target_column].eq(0)].copy()
    deaths = cohort.loc[cohort[target_column].eq(1)].copy()

    rows: List[List[str]] = []

    patient_id_column = first_existing_column(
        cohort,
        ["subject_id", "subjectid", "patient_id", "patientid"],
    )
    if patient_id_column is not None:
        rows.append(
            [
                "Unique patients, n",
                f"{cohort[patient_id_column].nunique(dropna=True):,}",
                f"{survivors[patient_id_column].nunique(dropna=True):,}",
                f"{deaths[patient_id_column].nunique(dropna=True):,}",
            ]
        )

    rows.append(
        [
            "Surgical encounters, n",
            f"{len(cohort):,}",
            f"{len(survivors):,}",
            f"{len(deaths):,}",
        ]
    )

    add_continuous_row(
        rows,
        cohort,
        survivors,
        deaths,
        ["age"],
        "Age, years",
    )

    sex_column = first_existing_column(cohort, ["sex", "gender"])
    if sex_column is not None:
        sex_group = normalize_sex(cohort[sex_column])
        valid = sex_group.ne("Missing/Other")
        rows.append(
            [
                "Male sex, n (%)",
                binary_summary(sex_group.eq("Male"), valid),
                binary_summary(
                    sex_group.loc[survivors.index].eq("Male"),
                    valid.loc[survivors.index],
                ),
                binary_summary(
                    sex_group.loc[deaths.index].eq("Male"),
                    valid.loc[deaths.index],
                ),
            ]
        )

    bmi_column = first_existing_column(cohort, ["bmi"])
    if bmi_column is not None:
        bmi = numeric(cohort[bmi_column]).where(
            numeric(cohort[bmi_column]).between(10, 80, inclusive="both")
        )
        rows.append(
            [
                "BMI, kg/m²",
                continuous_summary(bmi),
                continuous_summary(bmi.loc[survivors.index]),
                continuous_summary(bmi.loc[deaths.index]),
            ]
        )

    asa_column = first_existing_column(
        cohort,
        ["asa", "asa_status", "asa_class", "asa_ps"],
    )
    if asa_column is not None:
        asa = parse_asa(cohort[asa_column])
        for label, mask in (
            ("ASA I–II, n (%)", asa.isin([1, 2])),
            ("ASA III, n (%)", asa.eq(3)),
            ("ASA IV–V, n (%)", asa.isin([4, 5])),
            ("ASA missing/other, n (%)", ~asa.isin([1, 2, 3, 4, 5])),
        ):
            rows.append(
                [
                    label,
                    binary_summary(mask),
                    binary_summary(mask.loc[survivors.index]),
                    binary_summary(mask.loc[deaths.index]),
                ]
            )

    emergency_column = first_existing_column(
        cohort,
        ["emop", "emergency", "emergency_status", "urgency"],
    )
    if emergency_column is not None:
        emergency, valid = normalize_emergency(cohort[emergency_column])
        rows.append(
            [
                "Emergency surgery, n (%)",
                binary_summary(emergency, valid),
                binary_summary(
                    emergency.loc[survivors.index],
                    valid.loc[survivors.index],
                ),
                binary_summary(
                    emergency.loc[deaths.index],
                    valid.loc[deaths.index],
                ),
            ]
        )

    icu_flag, icu_valid, _ = derive_icu_status(cohort)
    if icu_flag is not None and icu_valid is not None:
        rows.append(
            [
                "Postoperative ICU admission, n (%)",
                binary_summary(icu_flag, icu_valid),
                binary_summary(
                    icu_flag.loc[survivors.index],
                    icu_valid.loc[survivors.index],
                ),
                binary_summary(
                    icu_flag.loc[deaths.index],
                    icu_valid.loc[deaths.index],
                ),
            ]
        )

    add_continuous_row(
        rows,
        cohort,
        survivors,
        deaths,
        ["or_duration"],
        "Operating-room duration, min",
    )
    add_continuous_row(
        rows,
        cohort,
        survivors,
        deaths,
        ["surgery_duration"],
        "Surgery duration, min",
    )
    add_continuous_row(
        rows,
        cohort,
        survivors,
        deaths,
        ["anesthesia_duration"],
        "Anesthesia duration, min",
    )

    return pd.DataFrame(
        rows,
        columns=[
            "Characteristic",
            "Overall",
            "Survivors",
            "Non-survivors",
        ],
    )


# =============================================================================
# MODEL METRIC HELPERS — TABLE 2 + TABLE S3
# =============================================================================

def format_estimate_ci(
    frame: pd.DataFrame,
    estimate_candidates: Sequence[str],
    low_candidates: Sequence[str],
    high_candidates: Sequence[str],
    digits: int = 3,
) -> pd.Series:
    estimate_col = first_existing_column(frame, estimate_candidates)
    low_col = first_existing_column(frame, low_candidates)
    high_col = first_existing_column(frame, high_candidates)

    if estimate_col is None:
        return pd.Series(["NA"] * len(frame), index=frame.index)

    estimates = pd.to_numeric(frame[estimate_col], errors="coerce")

    if low_col is None or high_col is None:
        return estimates.map(
            lambda x: "NA" if not np.isfinite(x) else f"{x:.{digits}f}"
        )

    lows = pd.to_numeric(frame[low_col], errors="coerce")
    highs = pd.to_numeric(frame[high_col], errors="coerce")

    output = []
    for estimate, low, high in zip(estimates, lows, highs):
        if not np.isfinite(estimate):
            output.append("NA")
        elif np.isfinite(low) and np.isfinite(high):
            output.append(
                f"{estimate:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"
            )
        else:
            output.append(f"{estimate:.{digits}f}")
    return pd.Series(output, index=frame.index)


def build_main_performance_table(
    metrics: pd.DataFrame,
    test_n: int,
) -> pd.DataFrame:
    model_col = first_existing_column(metrics, ["model"])
    if model_col is None:
        raise KeyError("test_model_metrics.csv is missing model.")

    output = pd.DataFrame()
    output["Model"] = metrics[model_col].astype(str)

    output["AUROC (95% CI)"] = format_estimate_ci(
        metrics,
        ["AUROC"],
        ["AUROC_CI_low", "AUROC_ci_lower_95", "AUROC_low"],
        ["AUROC_CI_high", "AUROC_ci_upper_95", "AUROC_high"],
        3,
    )
    output["AUPRC (95% CI)"] = format_estimate_ci(
        metrics,
        ["AUPRC"],
        ["AUPRC_CI_low", "AUPRC_ci_lower_95", "AUPRC_low"],
        ["AUPRC_CI_high", "AUPRC_ci_upper_95", "AUPRC_high"],
        3,
    )

    logical_columns = [
        ("Brier", ["Brier", "brier"]),
        ("Threshold", ["Threshold", "threshold"]),
        ("Sensitivity", ["Sensitivity", "sensitivity", "Recall", "recall"]),
        ("Specificity", ["Specificity", "specificity"]),
        ("PPV", ["PPV", "precision_PPV", "precision", "positive_predictive_value"]),
    ]

    for label, candidates in logical_columns:
        col = first_existing_column(metrics, candidates)
        if col is not None:
            output[label] = pd.to_numeric(metrics[col], errors="coerce")

    alert_col = first_existing_column(
        metrics,
        [
            "alert_rate",
            "alert_fraction",
            "alerted_fraction",
            "alert_percent",
            "alerted_percent",
        ],
    )

    if alert_col is not None:
        values = pd.to_numeric(metrics[alert_col], errors="coerce")
        if values.dropna().max() <= 1.0:
            values = 100.0 * values
        output["Alerts (%)"] = values
    else:
        tp_col = first_existing_column(metrics, ["TP", "tp"])
        fp_col = first_existing_column(metrics, ["FP", "fp"])
        if tp_col is not None and fp_col is not None and test_n > 0:
            output["Alerts (%)"] = (
                100.0
                * (
                    pd.to_numeric(metrics[tp_col], errors="coerce")
                    + pd.to_numeric(metrics[fp_col], errors="coerce")
                )
                / test_n
            )

    return output


def load_test_metrics_with_bootstrap_ci(
    run_dir: pathlib.Path,
    *,
    test_n: int,
) -> Tuple[pd.DataFrame, str]:
    """
    Load held-out metrics and merge the saved bootstrap confidence intervals.

    The final run stores point estimates in test_model_metrics.csv and bootstrap
    intervals in bootstrap_confidence_intervals.csv.  The publication tables
    must use both; otherwise a header such as "AUROC (95% CI)" would contain
    only a point estimate.
    """
    metrics_path = require_file(run_dir / "test_model_metrics.csv")
    bootstrap_path = require_file(
        run_dir / "bootstrap_confidence_intervals.csv"
    )

    metrics = pd.read_csv(metrics_path)
    bootstrap = pd.read_csv(bootstrap_path)

    required_bootstrap = {
        "model",
        "metric",
        "CI_low",
        "CI_high",
    }
    missing = required_bootstrap - set(bootstrap.columns)
    if missing:
        raise KeyError(
            "bootstrap_confidence_intervals.csv is missing: "
            f"{sorted(missing)}"
        )

    pivot_low = bootstrap.pivot_table(
        index="model",
        columns="metric",
        values="CI_low",
        aggfunc="first",
    )
    pivot_high = bootstrap.pivot_table(
        index="model",
        columns="metric",
        values="CI_high",
        aggfunc="first",
    )

    for metric in [
        "AUROC",
        "AUPRC",
        "Brier",
        "Accuracy",
        "Balanced_Accuracy",
        "Recall",
        "Specificity",
        "Precision",
    ]:
        if metric in pivot_low.columns:
            metrics[f"{metric}_CI_low"] = (
                metrics["model"].map(pivot_low[metric])
            )
            metrics[f"{metric}_CI_high"] = (
                metrics["model"].map(pivot_high[metric])
            )

    # Normalize operating-point aliases used by the Word tables.
    if "Sensitivity" not in metrics.columns and "Recall" in metrics.columns:
        metrics["Sensitivity"] = pd.to_numeric(
            metrics["Recall"],
            errors="coerce",
        )

    if "PPV" not in metrics.columns and "Precision" in metrics.columns:
        metrics["PPV"] = pd.to_numeric(
            metrics["Precision"],
            errors="coerce",
        )

    if "Sensitivity_CI_low" not in metrics.columns and "Recall_CI_low" in metrics.columns:
        metrics["Sensitivity_CI_low"] = metrics["Recall_CI_low"]
        metrics["Sensitivity_CI_high"] = metrics["Recall_CI_high"]

    if "PPV_CI_low" not in metrics.columns and "Precision_CI_low" in metrics.columns:
        metrics["PPV_CI_low"] = metrics["Precision_CI_low"]
        metrics["PPV_CI_high"] = metrics["Precision_CI_high"]

    # Alert burden is derived directly from the saved confusion counts if the
    # metrics file does not already contain it.
    if "Alerts (%)" not in metrics.columns:
        tp_col = first_existing_column(metrics, ["TP", "tp"])
        fp_col = first_existing_column(metrics, ["FP", "fp"])

        if tp_col is not None and fp_col is not None and test_n > 0:
            metrics["Alerts (%)"] = (
                100.0
                * (
                    pd.to_numeric(metrics[tp_col], errors="coerce")
                    + pd.to_numeric(metrics[fp_col], errors="coerce")
                )
                / float(test_n)
            )

    # Publication integrity: the final run is expected to contain bootstrap
    # AUROC/AUPRC intervals for every model.
    for metric in ["AUROC", "AUPRC"]:
        low = f"{metric}_CI_low"
        high = f"{metric}_CI_high"

        if low not in metrics.columns or high not in metrics.columns:
            raise KeyError(
                f"Missing merged {metric} confidence-interval columns."
            )

        if metrics[[low, high]].isna().any().any():
            missing_models = metrics.loc[
                metrics[[low, high]].isna().any(axis=1),
                "model",
            ].astype(str).tolist()

            raise ValueError(
                f"Missing {metric} bootstrap CI for model(s): "
                f"{missing_models}"
            )

    source = (
        f"{metrics_path} + {bootstrap_path}"
    )

    return metrics, source




# =============================================================================
# TABLE S2 — FEATURE SELECTION / DOMAIN COMPOSITION
# =============================================================================

def reconstruct_raw_predictor_count(
    reconstructed: Mapping[str, Any],
) -> Optional[int]:
    pipeline = reconstructed["pipeline"]
    cohort = reconstructed["cohort"]

    try:
        X_raw = pipeline.extract_model_features_classification(
            cohort,
            verbose=False,
        )
        X_raw = pipeline.clean_raw_predictors(X_raw)
        return int(X_raw.shape[1])
    except Exception as exc:
        print("Raw predictor count unavailable:", exc)
        return None


def build_feature_selection_tables(
    reconstructed: Mapping[str, Any],
    run_dir: pathlib.Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_predictor_count = reconstruct_raw_predictor_count(reconstructed)

    config = reconstructed["config"]
    n_folds = int(config.get("n_folds", 5))

    fold_rows: List[Dict[str, Any]] = []
    domain_rows: List[Dict[str, Any]] = []

    for fold_number in range(1, n_folds + 1):
        fold_dir = run_dir / "fold_artifacts" / f"fold_{fold_number}"
        if not fold_dir.exists():
            print(f"Warning: fold directory missing: {fold_dir}")
            continue

        selected_names_file = fold_dir / "selected_feature_names.npy"
        metadata_file = fold_dir / "selected_domain_metadata.json"

        if not selected_names_file.exists() or not metadata_file.exists():
            print(f"Warning: incomplete fold artifacts in {fold_dir}")
            continue

        selected_names = np.load(
            selected_names_file,
            allow_pickle=True,
        )
        metadata = load_json(metadata_file)

        transformed_count = np.nan
        variance_retained = np.nan

        preprocessor_file = fold_dir / "preprocessor.joblib"
        variance_file = fold_dir / "variance_selector.joblib"

        if preprocessor_file.exists():
            try:
                preprocessor = joblib.load(preprocessor_file)
                if hasattr(preprocessor, "get_feature_names_out"):
                    transformed_count = int(
                        len(preprocessor.get_feature_names_out())
                    )
            except Exception:
                pass

        if variance_file.exists():
            try:
                variance_selector = joblib.load(variance_file)
                if hasattr(variance_selector, "get_support"):
                    variance_retained = int(
                        np.asarray(
                            variance_selector.get_support(),
                            dtype=bool,
                        ).sum()
                    )
            except Exception:
                pass

        fold_rows.append(
            {
                "Fold": fold_number,
                "Raw_predictors": raw_predictor_count,
                "Transformed_features": transformed_count,
                "Variance_retained_features": variance_retained,
                "Selected_features": int(len(selected_names)),
            }
        )

        domain_indices = metadata.get("domain_indices", {})
        if isinstance(domain_indices, dict) and domain_indices:
            for domain, indices in domain_indices.items():
                domain_rows.append(
                    {
                        "Fold": fold_number,
                        "Domain": str(domain),
                        "Selected_features": int(len(indices)),
                    }
                )
        else:
            # Fallback if counts are saved instead of indices.
            domain_counts = metadata.get("domain_counts", {})
            for domain, count in domain_counts.items():
                domain_rows.append(
                    {
                        "Fold": fold_number,
                        "Domain": str(domain),
                        "Selected_features": int(count),
                    }
                )

    return pd.DataFrame(fold_rows), pd.DataFrame(domain_rows)


# =============================================================================
# MODEL / PROBABILITY RESOLUTION
# =============================================================================

def match_model_row(
    metrics: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    model_col = first_existing_column(metrics, ["model"])
    if model_col is None:
        raise KeyError("Model metrics file is missing model.")

    normalized = metrics[model_col].astype(str).map(normalize_name)
    target = normalize_name(model_name)

    exact = metrics.loc[normalized.eq(target)]
    if not exact.empty:
        return exact.iloc[0]

    aliases = MODEL_ALIASES.get(model_name, [target])
    for alias in aliases:
        match = metrics.loc[
            normalized.str.contains(alias, regex=False)
        ]
        if not match.empty:
            return match.iloc[0]

    raise KeyError(
        f"Model '{model_name}' was not found. Available models: "
        f"{metrics[model_col].astype(str).tolist()}"
    )


def resolve_probability_column(
    predictions: pd.DataFrame,
    model_name: str,
) -> str:
    candidates = [
        str(column)
        for column in predictions.columns
        if (
            "prob" in str(column).lower()
            or "risk" in str(column).lower()
        )
    ]
    if not candidates:
        raise KeyError("No probability-like columns found in test predictions.")

    aliases = MODEL_ALIASES.get(
        model_name,
        [normalize_name(model_name)],
    )

    scored: List[Tuple[float, str]] = []

    for column in candidates:
        norm = normalize_name(column)
        score = 0.0

        for alias in aliases:
            if alias in norm:
                score += 20.0 + len(alias) / 100.0

        if "calibrated" in norm:
            score += 5.0
        if "probability" in norm:
            score += 2.0
        if norm.startswith("prob"):
            score += 1.0

        if (
            model_name == "Hybrid DARN-XGB Blend"
            and "hybrid" not in norm
            and "blend" not in norm
        ):
            score -= 10.0

        scored.append((score, column))

    scored.sort(reverse=True)
    best_score, best_column = scored[0]

    if best_score <= 0:
        raise KeyError(
            f"Could not resolve probability column for '{model_name}'. "
            f"Candidates: {candidates}"
        )

    return best_column


def resolve_audit_model(
    run_dir: pathlib.Path,
    predictions: pd.DataFrame,
    requested_model: str,
) -> Dict[str, Any]:
    metrics = pd.read_csv(require_file(run_dir / "test_model_metrics.csv"))

    model_name = (
        DEFAULT_AUDIT_MODEL
        if str(requested_model).strip().lower() in {"", "auto"}
        else str(requested_model)
    )

    model_row = match_model_row(metrics, model_name)
    actual_model = str(
        model_row[
            first_existing_column(metrics, ["model"])
        ]
    )

    threshold_col = first_existing_column(
        metrics,
        ["Threshold", "threshold"],
    )
    if threshold_col is None:
        raise KeyError("test_model_metrics.csv has no threshold column.")

    probability_col = resolve_probability_column(
        predictions,
        actual_model,
    )

    probability = pd.to_numeric(
        predictions[probability_col],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(probability).all():
        raise ValueError("Audit-model probabilities contain non-finite values.")

    return {
        "metrics": metrics,
        "model_name": actual_model,
        "model_row": model_row,
        "threshold": float(model_row[threshold_col]),
        "probability_column": probability_col,
        "probability": probability,
    }


# =============================================================================
# TABLE S7 — SUBGROUP AUDIT
# =============================================================================

def derive_subgroups(test_cohort: pd.DataFrame) -> Dict[str, pd.Series]:
    groups: Dict[str, pd.Series] = {}

    sex_col = first_existing_column(test_cohort, ["sex", "gender"])
    if sex_col is not None:
        groups["Sex"] = normalize_sex(test_cohort[sex_col])

    age_col = first_existing_column(test_cohort, ["age"])
    if age_col is not None:
        age = pd.to_numeric(test_cohort[age_col], errors="coerce")
        age_group = pd.Series(
            "Missing/Other",
            index=test_cohort.index,
            dtype=object,
        )
        age_group.loc[age.lt(60)] = "<60"
        age_group.loc[age.ge(60) & age.lt(70)] = "60–69"
        age_group.loc[age.ge(70)] = "≥70"
        groups["Age"] = age_group

    asa_col = first_existing_column(
        test_cohort,
        ["asa", "asa_class", "asa_status", "asa_ps"],
    )
    if asa_col is not None:
        asa = parse_asa(test_cohort[asa_col])
        asa_group = pd.Series(
            "Missing/Other",
            index=test_cohort.index,
            dtype=object,
        )
        asa_group.loc[asa.isin([1, 2])] = "I–II"
        asa_group.loc[asa.eq(3)] = "III"
        asa_group.loc[asa.isin([4, 5])] = "IV–V"
        groups["ASA"] = asa_group

    urgency_col = first_existing_column(
        test_cohort,
        ["emop", "emergency", "emergency_status", "urgency", "urgent"],
    )
    if urgency_col is not None:
        flag, valid = normalize_emergency(test_cohort[urgency_col])
        urgency = pd.Series(
            "Missing/Other",
            index=test_cohort.index,
            dtype=object,
        )
        urgency.loc[valid & ~flag] = "Elective"
        urgency.loc[valid & flag] = "Emergency"
        groups["Urgency"] = urgency

    return groups


def point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    predicted = (p >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))

    result = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "mean_predicted_risk": float(p.mean()) if len(p) else np.nan,
        "Brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "PPV": safe_divide(tp, tp + fp),
        "NPV": safe_divide(tn, tn + fn),
        "FPR": safe_divide(fp, fp + tn),
        "observed_expected_ratio": safe_divide(
            float(y.sum()),
            float(p.sum()),
        ),
    }

    if len(np.unique(y)) == 2:
        result["AUROC"] = float(roc_auc_score(y, p))
        result["AUPRC"] = float(average_precision_score(y, p))
    else:
        result["AUROC"] = np.nan
        result["AUPRC"] = np.nan

    return result


def bootstrap_group_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    seed: int,
    n_jobs: int,
) -> Dict[str, Tuple[float, float]]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    metric_names = [
        "AUROC",
        "AUPRC",
        "Brier",
        "sensitivity",
        "specificity",
        "PPV",
        "FPR",
        "observed_expected_ratio",
    ]

    seed_sequence = np.random.SeedSequence(seed)
    replicate_seeds = [
        int(child.generate_state(1)[0])
        for child in seed_sequence.spawn(n_bootstrap)
    ]

    def one(rep_seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(rep_seed)

        if len(positive) and len(negative):
            indices = np.concatenate(
                [
                    rng.choice(
                        positive,
                        size=len(positive),
                        replace=True,
                    ),
                    rng.choice(
                        negative,
                        size=len(negative),
                        replace=True,
                    ),
                ]
            )
            rng.shuffle(indices)
        else:
            indices = rng.integers(0, len(y), size=len(y))

        return point_metrics(
            y[indices],
            p[indices],
            threshold,
        )

    results = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        verbose=0,
    )(
        delayed(one)(rep_seed)
        for rep_seed in replicate_seeds
    )

    output: Dict[str, Tuple[float, float]] = {}

    for metric in metric_names:
        values = np.asarray(
            [result.get(metric, np.nan) for result in results],
            dtype=float,
        )
        values = values[np.isfinite(values)]

        if len(values):
            output[metric] = (
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            )
        else:
            output[metric] = (np.nan, np.nan)

    return output


def build_subgroup_table(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    groups: Mapping[str, pd.Series],
    *,
    n_bootstrap: int,
    n_jobs: int,
    random_state: int,
) -> pd.DataFrame:
    definitions: List[Tuple[str, str, np.ndarray]] = [
        (
            "Overall",
            "Overall",
            np.ones(len(y), dtype=bool),
        )
    ]

    for variable, series in groups.items():
        series = series.reset_index(drop=True)
        for level in pd.unique(series):
            if str(level) == "Missing/Other":
                continue
            definitions.append(
                (
                    variable,
                    str(level),
                    series.eq(level).to_numpy(),
                )
            )

    rows: List[Dict[str, Any]] = []

    for index, (variable, level, mask) in enumerate(definitions):
        yg = y[mask]
        pg = p[mask]

        if len(yg) < 30:
            continue

        point = point_metrics(yg, pg, threshold)
        intervals = bootstrap_group_metrics(
            yg,
            pg,
            threshold,
            n_bootstrap=n_bootstrap,
            seed=random_state + 1009 * (index + 1),
            n_jobs=n_jobs,
        )

        row: Dict[str, Any] = {
            "subgroup_variable": variable,
            "subgroup_level": level,
            **point,
        }

        for metric, (low, high) in intervals.items():
            row[f"{metric}_ci_lower_95"] = low
            row[f"{metric}_ci_upper_95"] = high

        rows.append(row)

    table = pd.DataFrame(rows)

    order_map = {
        pair: order
        for order, pair in enumerate(DISPLAY_ORDER)
    }

    table["display_order"] = [
        order_map.get(
            (row.subgroup_variable, row.subgroup_level),
            999,
        )
        for row in table.itertuples()
    ]

    table["display_label"] = table.apply(
        lambda row: (
            "Overall"
            if row["subgroup_variable"] == "Overall"
            else (
                f"{row['subgroup_variable']}: "
                f"{row['subgroup_level']}"
            )
        ),
        axis=1,
    )

    return (
        table.sort_values(
            ["display_order", "subgroup_variable", "subgroup_level"]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# TABLE S8 — TAIL CALIBRATION
# =============================================================================

def wilson_interval(
    events: int,
    n: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan

    proportion = events / n
    denominator = 1.0 + z**2 / n
    center = (
        proportion + z**2 / (2.0 * n)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + z**2 / (4.0 * n**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def build_tail_calibration_table(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    ranks = pd.Series(p).rank(method="first")

    labels = pd.qcut(
        ranks,
        q=min(n_bins, len(ranks)),
        labels=False,
        duplicates="drop",
    )

    frame = pd.DataFrame(
        {
            "y_true": y,
            "probability": p,
            "risk_bin": labels.astype(int) + 1,
        }
    )

    rows: List[Dict[str, Any]] = []

    for risk_bin, subset in frame.groupby(
        "risk_bin",
        observed=True,
    ):
        n = int(len(subset))
        events = int(subset["y_true"].sum())
        low, high = wilson_interval(events, n)

        rows.append(
            {
                "risk_bin": int(risk_bin),
                "n": n,
                "events": events,
                "mean_predicted_risk": float(
                    subset["probability"].mean()
                ),
                "median_predicted_risk": float(
                    subset["probability"].median()
                ),
                "minimum_predicted_risk": float(
                    subset["probability"].min()
                ),
                "maximum_predicted_risk": float(
                    subset["probability"].max()
                ),
                "observed_mortality": float(
                    subset["y_true"].mean()
                ),
                "observed_ci_lower_95": low,
                "observed_ci_upper_95": high,
            }
        )

    return pd.DataFrame(rows).sort_values("risk_bin").reset_index(drop=True)


# =============================================================================
# TABLES S9-S10 — ERROR PROFILE + FALSE-NEGATIVE REVIEW
# =============================================================================

def assign_error_groups(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> np.ndarray:
    predicted = (p >= threshold).astype(int)

    return np.select(
        [
            (y == 1) & (predicted == 1),
            (y == 0) & (predicted == 1),
            (y == 1) & (predicted == 0),
            (y == 0) & (predicted == 0),
        ],
        ERROR_GROUP_ORDER,
        default="Unknown",
    )


def build_error_profile(
    test_cohort: pd.DataFrame,
    error_groups: np.ndarray,
    p: np.ndarray,
    groups: Mapping[str, pd.Series],
) -> pd.DataFrame:
    age_group = groups.get(
        "Age",
        pd.Series("Missing/Other", index=test_cohort.index),
    )
    asa_group = groups.get(
        "ASA",
        pd.Series("Missing/Other", index=test_cohort.index),
    )
    urgency_group = groups.get(
        "Urgency",
        pd.Series("Missing/Other", index=test_cohort.index),
    )

    rows: List[Dict[str, Any]] = []

    for group in ERROR_GROUP_ORDER:
        mask = error_groups == group
        n_group = int(mask.sum())

        rows.append(
            {
                "error_group": group,
                "n": n_group,
                "Age_70_or_older_percent": (
                    100.0 * float(age_group.loc[mask].eq("≥70").mean())
                    if n_group
                    else np.nan
                ),
                "ASA_III_to_V_percent": (
                    100.0
                    * float(
                        asa_group.loc[mask]
                        .isin(["III", "IV–V"])
                        .mean()
                    )
                    if n_group
                    else np.nan
                ),
                "Emergency_percent": (
                    100.0
                    * float(
                        urgency_group.loc[mask]
                        .eq("Emergency")
                        .mean()
                    )
                    if n_group
                    else np.nan
                ),
                "median_predicted_risk": (
                    float(np.median(p[mask]))
                    if n_group
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_false_negative_review(
    test_cohort: pd.DataFrame,
    p: np.ndarray,
    threshold: float,
    error_groups: np.ndarray,
    groups: Mapping[str, pd.Series],
) -> pd.DataFrame:
    positions = np.where(error_groups == "False negative")[0]

    probability_percentile = pd.Series(p).rank(
        method="average",
        pct=True,
    ).to_numpy()

    candidate_columns: Dict[str, Sequence[str]] = {
        "age": ["age"],
        "sex_raw": ["sex", "gender"],
        "asa_raw": ["asa", "asa_class", "asa_status", "asa_ps"],
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
        "surgery_duration": ["surgery_duration"],
        "anesthesia_duration": ["anesthesia_duration"],
        "or_duration": ["or_duration"],
        "blood_loss": ["intraop_mean_ebl", "mean_ebl", "ebl"],
        "fibrinogen": ["preop_fibrinogen", "fibrinogen"],
        "platelets": [
            "preop_platelet",
            "preop_platelets",
            "platelet",
            "platelets",
        ],
        "glucose": ["preop_glucose", "glucose"],
    }

    resolved: Dict[str, str] = {}

    for output_name, candidates in candidate_columns.items():
        column = first_existing_column(test_cohort, candidates)
        if column is not None:
            resolved[output_name] = column

    diagnosis_columns = [
        column
        for column in test_cohort.columns
        if str(column).upper().startswith("DIAG_")
    ]

    rows: List[Dict[str, Any]] = []

    for position in positions:
        row: Dict[str, Any] = {
            "test_row_position": int(position),
            "predicted_probability": float(p[position]),
            "threshold": float(threshold),
            "distance_below_threshold": float(
                threshold - p[position]
            ),
            "predicted_risk_percentile": float(
                probability_percentile[position]
            ),
            "age_group": (
                str(groups["Age"].iloc[position])
                if "Age" in groups
                else "Unavailable"
            ),
            "sex_group": (
                str(groups["Sex"].iloc[position])
                if "Sex" in groups
                else "Unavailable"
            ),
            "asa_group": (
                str(groups["ASA"].iloc[position])
                if "ASA" in groups
                else "Unavailable"
            ),
            "urgency_group": (
                str(groups["Urgency"].iloc[position])
                if "Urgency" in groups
                else "Unavailable"
            ),
            "raw_missing_count": int(
                test_cohort.iloc[position].isna().sum()
            ),
            "raw_missing_fraction": float(
                test_cohort.iloc[position].isna().mean()
            ),
        }

        for output_name, column in resolved.items():
            row[output_name] = test_cohort.iloc[position][column]

        present_diagnoses: List[str] = []

        for diagnosis_column in diagnosis_columns:
            value = pd.to_numeric(
                pd.Series(
                    [test_cohort.iloc[position][diagnosis_column]]
                ),
                errors="coerce",
            ).iloc[0]

            if np.isfinite(value) and value > 0:
                present_diagnoses.append(str(diagnosis_column))

        row["present_diagnoses"] = " | ".join(
            present_diagnoses[:20]
        )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SAVED-ARTIFACT TABLE DISCOVERY
# =============================================================================

def read_best_masking_table(run_dir: pathlib.Path) -> Optional[pd.DataFrame]:
    candidates = [
        run_dir / "darn_input_domain_masking.csv",
        run_dir / "darn_domain_ablation.csv",
        run_dir / "tables" / "darn_input_domain_masking.csv",
        run_dir / "tables" / "darn_domain_ablation.csv",
    ]

    path = first_existing_file(candidates)

    if path is None:
        discovered = discover_csvs(
            run_dir,
            ["mask"],
            ["leave_one_domain", "eyeear"],
        )
        if discovered:
            path = discovered[0]

    if path is None:
        return None

    frame = pd.read_csv(path)
    frame = frame.copy()
    if "source_file" in frame.columns:
        frame["source_file"] = str(path)
    else:
        frame.insert(0, "source_file", str(path))
    return frame


def read_paired_model_comparisons(
    run_dir: pathlib.Path,
) -> Optional[pd.DataFrame]:
    """
    Read ONLY the canonical overall model-comparison paired-bootstrap artifacts
    for Table S4.

    Important:
      - Do not recurse through the entire run directory.
      - Eye/Ear sensitivity, LODO retraining, figure-derived duplicates, and
        publication-package outputs are NOT Table S4 sources.
      - Normalize heterogeneous source schemas into one publication-ready
        dataframe before saving Table S4.
    """

    preferred_paths = [
        run_dir / "paired_bootstrap_hybrid_vs_all.csv",
        run_dir / "paired_bootstrap_darn_vs_baselines.csv",
    ]

    source_paths = [
        path
        for path in preferred_paths
        if path.exists()
    ]

    # Conservative fallback: root-level paired-bootstrap CSVs only.
    # Never recurse into sensitivity/retraining/figure subdirectories.
    if not source_paths:
        for path in sorted(run_dir.glob("*.csv")):
            lower = path.name.lower()

            if "paired" not in lower or "bootstrap" not in lower:
                continue

            if any(
                blocked in lower
                for blocked in [
                    "eye",
                    "ear",
                    "lodo",
                    "leave_one_domain",
                    "feature_removal",
                    "retraining",
                    "figure",
                ]
            ):
                continue

            source_paths.append(path)

    if not source_paths:
        return None

    def get_col(
        frame: pd.DataFrame,
        candidates: Sequence[str],
    ) -> Optional[str]:
        return first_existing_column(frame, candidates)

    normalized_frames: List[pd.DataFrame] = []

    for path in source_paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        reference_col = get_col(
            frame,
            [
                "reference_model",
                "reference",
                "model_a",
                "model_1",
                "full_model",
                "model",
            ],
        )
        comparator_col = get_col(
            frame,
            [
                "comparator",
                "comparison_model",
                "model_b",
                "model_2",
                "comparison",
            ],
        )
        metric_col = get_col(
            frame,
            [
                "metric",
                "metric_name",
                "measure",
                "performance_metric",
            ],
        )
        difference_col = get_col(
            frame,
            [
                "difference_favoring_reference",
                "difference",
                "diff",
                "delta",
                "mean_difference",
                "estimate_difference",
            ],
        )
        ci_low_col = get_col(
            frame,
            [
                "CI_low",
                "ci_low",
                "ci_lower",
                "ci_lower_95",
                "lower_95",
                "difference_ci_lower_95",
            ],
        )
        ci_high_col = get_col(
            frame,
            [
                "CI_high",
                "ci_high",
                "ci_upper",
                "ci_upper_95",
                "upper_95",
                "difference_ci_upper_95",
            ],
        )
        p_col = get_col(
            frame,
            [
                "bootstrap_p_two_sided",
                "bootstrap_p",
                "paired_p",
                "p_value",
                "pvalue",
                "p",
            ],
        )
        n_boot_col = get_col(
            frame,
            [
                "n_bootstrap",
                "bootstrap_replicates",
                "n_boot",
            ],
        )

        required = [
            reference_col,
            comparator_col,
            metric_col,
            difference_col,
            ci_low_col,
            ci_high_col,
            p_col,
        ]

        if any(column is None for column in required):
            # This file is not a usable overall paired model-comparison table.
            continue

        out = pd.DataFrame(
            {
                "Reference model": frame[reference_col],
                "Comparator": frame[comparator_col],
                "Metric": frame[metric_col],
                "Difference": pd.to_numeric(
                    frame[difference_col],
                    errors="coerce",
                ),
                "CI low": pd.to_numeric(
                    frame[ci_low_col],
                    errors="coerce",
                ),
                "CI high": pd.to_numeric(
                    frame[ci_high_col],
                    errors="coerce",
                ),
                "p-value": pd.to_numeric(
                    frame[p_col],
                    errors="coerce",
                ),
            }
        )

        if n_boot_col is not None:
            out["Bootstrap replicates"] = pd.to_numeric(
                frame[n_boot_col],
                errors="coerce",
            )
        else:
            out["Bootstrap replicates"] = np.nan

        out["Source artifact"] = path.name

        # Manuscript Table S4 reports discrimination/error metrics only.
        metric_normalized = (
            out["Metric"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        out = out[
            metric_normalized.isin(
                {
                    "AUROC",
                    "AUPRC",
                    "BRIER",
                }
            )
        ].copy()

        # Remove rows without actual statistical results.
        out = out.dropna(
            subset=[
                "Difference",
                "CI low",
                "CI high",
                "p-value",
            ]
        )

        # Remove incomplete labels.
        out = out[
            out["Reference model"].notna()
            & out["Comparator"].notna()
            & out["Metric"].notna()
        ].copy()

        if not out.empty:
            normalized_frames.append(out)

    if not normalized_frames:
        return None

    result = pd.concat(
        normalized_frames,
        ignore_index=True,
        sort=False,
    )

    # Remove exact duplicate directed comparisons.
    result = result.drop_duplicates(
        subset=[
            "Reference model",
            "Comparator",
            "Metric",
        ],
        keep="first",
    ).reset_index(drop=True)

    # If the same pair appears in both directions, keep one deterministic
    # orientation. Prefer Hybrid Blend as reference, then Uniform DARN.
    preferred_reference_order = {
        "Hybrid DARN-XGB Blend": 0,
        "Uniform DARN": 1,
        "Hybrid DARN-XGB Stack": 2,
        "XGBoost": 3,
        "MLP": 4,
        "Logistic Regression": 5,
        "ASA-only Logistic": 6,
    }

    result["_reference_priority"] = (
        result["Reference model"]
        .map(preferred_reference_order)
        .fillna(99)
        .astype(int)
    )

    result["_pair_key"] = result.apply(
        lambda row: "||".join(
            sorted(
                [
                    str(row["Reference model"]),
                    str(row["Comparator"]),
                ]
            )
        ),
        axis=1,
    )

    result = (
        result.sort_values(
            [
                "_pair_key",
                "Metric",
                "_reference_priority",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=[
                "_pair_key",
                "Metric",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_reference_priority",
                "_pair_key",
            ]
        )
        .reset_index(drop=True)
    )

    # Stable publication ordering.
    metric_order = {
        "AUROC": 0,
        "AUPRC": 1,
        "BRIER": 2,
    }

    result["_metric_order"] = (
        result["Metric"]
        .astype(str)
        .str.upper()
        .map(metric_order)
        .fillna(99)
    )

    result["_reference_order"] = (
        result["Reference model"]
        .map(preferred_reference_order)
        .fillna(99)
    )

    result = (
        result.sort_values(
            [
                "_reference_order",
                "Comparator",
                "_metric_order",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_metric_order",
                "_reference_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result



def read_lodo_tables(
    run_dir: pathlib.Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    directory = run_dir / "leave_one_domain_out_retraining"

    metrics_path = first_existing_file(
        [
            directory / "leave_one_domain_out_retrained_test_metrics.csv",
            directory / "leave_one_domain_out_test_metrics.csv",
        ]
    )
    paired_path = first_existing_file(
        [
            directory / "leave_one_domain_out_paired_bootstrap_differences.csv",
            directory / "leave_one_domain_out_paired_bootstrap.csv",
        ]
    )

    metrics = (
        pd.read_csv(metrics_path)
        if metrics_path is not None
        else None
    )
    paired = (
        pd.read_csv(paired_path)
        if paired_path is not None
        else None
    )

    return metrics, paired


def read_explainability_table(
    run_dir: pathlib.Path,
    top_k: int,
) -> Optional[pd.DataFrame]:
    figure4_dir = (
        run_dir
        / "figures_revised_manuscript"
        / "figure4_darn_xgboost_AB_strict_oof"
    )

    darn_path = figure4_dir / "uniform_darn_gradientshap_feature_importance.csv"
    xgb_path = figure4_dir / "xgboost_treeshap_feature_importance.csv"

    frames: List[pd.DataFrame] = []

    if darn_path.exists():
        darn = pd.read_csv(darn_path)
        darn.insert(0, "component", "Uniform DARN")
        darn.insert(1, "attribution_method", "GradientSHAP")
        frames.append(darn.head(top_k))

    if xgb_path.exists():
        xgb = pd.read_csv(xgb_path)
        xgb.insert(0, "component", "XGBoost")
        xgb.insert(1, "attribution_method", "TreeSHAP")
        frames.append(xgb.head(top_k))

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True, sort=False)


def copy_eyeear_tables(
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
    manifest: List[Dict[str, Any]],
) -> None:
    """
    Copy only manuscript-facing Eye/Ear sensitivity-analysis tables.

    Row-level probability/prediction artifacts are deliberately excluded from
    the publication supplement. They remain in the run directory for audit
    and reproducibility.
    """
    candidates: List[pathlib.Path] = []

    for path in run_dir.rglob("*.csv"):
        # Do not recursively ingest Table S12 files created by an earlier run.
        if is_generated_publication_artifact(path):
            continue

        lower = path.name.lower()

        if not (
            "eyeear" in lower
            or "eye_ear" in lower
            or "eye-ear" in lower
        ):
            continue

        # Never promote row-level predictions/probabilities to a manuscript
        # supplementary table.
        if "probabilit" in lower or "prediction" in lower:
            continue

        candidates.append(path)

    if not candidates:
        add_manifest_entry(
            manifest,
            "Table S12",
            "Eye/Ear sensitivity analysis",
            None,
            "No manuscript-facing Eye/Ear CSV artifact discovered",
            status="not found",
        )
        return

    def classify(path: pathlib.Path) -> Tuple[int, str, str]:
        lower = path.name.lower()

        if "adjusted_association" in lower:
            return 1, "Table S12a", "Eye/Ear adjusted association"

        if (
            "paired_bootstrap" in lower
            or "paired" in lower
            and "bootstrap" in lower
        ):
            return (
                3,
                "Table S12c",
                "Eye/Ear feature-removal paired-bootstrap comparison",
            )

        if "test_metrics" in lower or "metrics" in lower:
            return 2, "Table S12b", "Eye/Ear feature-removal test metrics"

        return 9, "Table S12", f"Eye/Ear sensitivity analysis: {path.stem}"

    ordered = sorted(
        set(candidates),
        key=lambda path: classify(path)[0:2],
    )

    used_ids = Counter()

    for path in ordered:
        _, base_id, title = classify(path)
        used_ids[base_id] += 1

        table_id = (
            base_id
            if used_ids[base_id] == 1
            else f"{base_id}.{used_ids[base_id]}"
        )

        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            path.name,
        )

        destination = output_dir / f"{table_id.replace(' ', '')}_{safe_name}"
        shutil.copy2(path, destination)

        add_manifest_entry(
            manifest,
            table_id,
            title,
            destination,
            str(path),
        )



# =============================================================================
# PUBLICATION DISPLAY / DECIMAL POLICY
# =============================================================================
#
# IMPORTANT:
# - Raw CSV outputs retain their original numerical precision.
# - Rounding is applied ONLY when values are rendered into the Word tables.
# - This prevents loss of precision in the machine-readable analysis artifacts.

def _fmt_number(value: Any, digits: int) -> str:
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return f"{number:.{digits}f}"


def _fmt_integer(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return f"{int(round(number)):,}"


def _fmt_percent_fraction(
    value: Any,
    digits: int = 1,
) -> str:
    """Format a 0–1 proportion as a percentage."""
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return f"{100.0 * number:.{digits}f}%"


def _fmt_percent_already(
    value: Any,
    digits: int = 1,
) -> str:
    """Format a value already expressed on a 0–100 percent scale."""
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return f"{number:.{digits}f}%"


def _fmt_probability_percent(
    value: Any,
    digits: int = 3,
) -> str:
    """Format a low absolute probability as percent for readability."""
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return f"{100.0 * number:.{digits}f}%"


def _fmt_pvalue(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "" if pd.isna(value) else str(value)

    if not np.isfinite(number):
        return ""

    return "<0.001" if number < 0.001 else f"{number:.3f}"


def _fmt_or_hr(value: Any) -> str:
    return _fmt_number(value, 2)


def _normalize_column_token(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def _row_metric_name(
    frame: pd.DataFrame,
    row_index: Any,
) -> str:
    """
    Resolve the metric represented by a row in paired-comparison tables.

    Supports tables that store AUROC/AUPRC/Brier as row values rather than
    separate columns.
    """
    for candidate in [
        "metric",
        "metric_name",
        "measure",
        "performance_metric",
    ]:
        column = first_existing_column(frame, [candidate])
        if column is not None:
            return normalize_name(frame.loc[row_index, column])

    return ""


def _format_paired_difference_value(
    value: Any,
    metric_name: str,
) -> str:
    """
    Paired differences need more precision than absolute model metrics.
    """
    if "brier" in metric_name:
        return _fmt_number(value, 6)

    if "auroc" in metric_name or "auprc" in metric_name:
        return _fmt_number(value, 4)

    return _fmt_number(value, 4)


def format_publication_dataframe(
    frame: pd.DataFrame,
    table_id: str,
) -> pd.DataFrame:
    """
    Return a display-only copy using the locked manuscript decimal policy.

    Raw CSVs remain unchanged. This function is used only by Word rendering.
    """
    display = frame.copy()
    table_token = normalize_name(table_id)

    # Table 1 is already constructed as manuscript-ready strings:
    # continuous variables use 1 decimal and n (%) uses 1 decimal percent.
    if table_token == normalize_name("Table 1"):
        return display

    # ------------------------------------------------------------------
    # Main Table 2 — compact clinical display.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table 2"):
        for column in display.columns:
            token = _normalize_column_token(column)

            if token == "brier":
                display[column] = display[column].map(
                    lambda x: _fmt_number(x, 5)
                )

            elif token == "threshold":
                display[column] = display[column].map(
                    lambda x: _fmt_number(x, 5)
                )

            elif token in {
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "fpr",
                "fnr",
            }:
                display[column] = display[column].map(
                    lambda x: _fmt_percent_fraction(x, 1)
                )

            elif token in {
                "alerts",
                "alerts_percent",
                "alert_percent",
                "alerted_percent",
            } or "alerts" in token:
                display[column] = display[column].map(
                    lambda x: _fmt_percent_already(x, 1)
                )

        return display

    # ------------------------------------------------------------------
    # Paired model-comparison / LODO paired-bootstrap tables.
    #
    # These artifacts use several naming conventions across analyses. We format
    # numeric statistical columns from the row's Metric value rather than
    # depending only on exact column aliases.
    # ------------------------------------------------------------------
    if table_token in {
        normalize_name("Table S4"),
        normalize_name("Table S6b"),
        normalize_name("Table S12c"),
    }:
        label_tokens = {
            "model",
            "model_a",
            "model_b",
            "comparison",
            "contrast",
            "removed_domain",
            "domain_removed",
            "domain",
            "metric",
            "metric_name",
            "measure",
            "performance_metric",
            "source_file",
            "analysis_method",
            "reference_model",
            "comparison_model",
        }

        absolute_value_hints = [
            "model_a",
            "model_b",
            "value_a",
            "value_b",
            "estimate_a",
            "estimate_b",
            "complete",
            "removed",
            "full",
            "reduced",
            "baseline",
            "reference",
            "comparison_value",
        ]

        for row_index in display.index:
            metric_name = _row_metric_name(display, row_index)

            for column in display.columns:
                token = _normalize_column_token(column)

                if token in label_tokens:
                    continue

                value = display.at[row_index, column]

                # Leave genuinely textual cells unchanged.
                try:
                    number = float(value)
                except Exception:
                    continue

                if not np.isfinite(number):
                    display.at[row_index, column] = ""
                    continue

                # Counts / bootstrap iterations.
                if any(
                    hint in token
                    for hint in [
                        "n_boot",
                        "bootstrap_replicates",
                        "replicates",
                        "iterations",
                    ]
                ):
                    display.at[row_index, column] = _fmt_integer(number)
                    continue

                # P-values.
                if (
                    token == "p"
                    or token.endswith("_p")
                    or "p_value" in token
                    or "pvalue" in token
                ):
                    display.at[row_index, column] = _fmt_pvalue(number)
                    continue

                is_absolute_value = any(
                    hint in token
                    for hint in absolute_value_hints
                ) and not any(
                    hint in token
                    for hint in [
                        "diff",
                        "difference",
                        "delta",
                        "minus",
                        "change",
                    ]
                )

                if "brier" in metric_name:
                    digits = 5 if is_absolute_value else 6
                elif (
                    "auroc" in metric_name
                    or "auprc" in metric_name
                ):
                    digits = 3 if is_absolute_value else 4
                elif (
                    "absolute_oe_error" in metric_name
                    or "absolute o:e error" in metric_name
                    or "absolute oe error" in metric_name
                ):
                    digits = 4 if is_absolute_value else 5
                else:
                    digits = 4

                display.at[row_index, column] = _fmt_number(
                    number,
                    digits,
                )

        return display

    # ------------------------------------------------------------------
    # Generic supplementary-table display rules.
    #
    # Match semantic substrings rather than only exact column names because
    # analysis artifacts may use names such as baseline_AUROC,
    # sensitivity_ci_lower_95, mean_absolute_direct_attribution, etc.
    # ------------------------------------------------------------------
    for column in display.columns:
        token = _normalize_column_token(column)

        # Counts / ranks / bin identifiers.
        if token in {
            "n",
            "events",
            "deaths",
            "encounters",
            "tp",
            "fp",
            "tn",
            "fn",
            "alerted_n",
            "captured_deaths_n",
            "selected_features",
            "raw_predictors",
            "transformed_features",
            "variance_retained_features",
            "bootstrap_replicates",
            "bootstrap_replicates_per_subgroup",
            "rank",
            "risk_bin",
            "test_row_position",
            "holdout_row_position",
            "raw_missing_count",
        }:
            display[column] = display[column].map(_fmt_integer)
            continue

        # P-values: robust to names such as paired_bootstrap_p_value.
        if (
            token in {
                "p",
                "p_value",
                "pvalue",
                "wald_p",
                "lr_p",
                "interaction_wald_p",
                "interaction_lr_p",
                "paired_p",
                "bootstrap_p",
            }
            or token.endswith("_p")
            or token.endswith("_p_value")
            or token.endswith("_pvalue")
            or "p_value" in token
        ):
            display[column] = display[column].map(_fmt_pvalue)
            continue

        # Absolute and prefixed/suffixed AUROC/AUPRC columns.
        if "auroc" in token or "auprc" in token:
            is_difference = any(
                term in token
                for term in [
                    "diff",
                    "difference",
                    "delta",
                    "drop",
                    "change",
                    "minus",
                ]
            )

            digits = 4 if is_difference else 3
            display[column] = display[column].map(
                lambda x, d=digits: _fmt_number(x, d)
            )
            continue

        # Brier: absolute values 5 decimals; differences/deltas 6.
        if "brier" in token:
            is_difference = any(
                term in token
                for term in [
                    "diff",
                    "difference",
                    "delta",
                    "drop",
                    "change",
                    "minus",
                ]
            )

            digits = 6 if is_difference else 5
            display[column] = display[column].map(
                lambda x, d=digits: _fmt_number(x, d)
            )
            continue

        # Calibration error and calibration model parameters.
        if (
            token == "ece"
            or token.endswith("_ece")
            or "expected_calibration_error" in token
        ):
            display[column] = display[column].map(
                lambda x: _fmt_number(x, 5)
            )
            continue

        if any(
            key in token
            for key in [
                "calibration_slope",
                "calibration_intercept",
                "observed_expected_ratio",
                "oe_ratio",
                "o_e_ratio",
            ]
        ):
            display[column] = display[column].map(
                lambda x: _fmt_number(x, 3)
            )
            continue

        # Absolute O:E error / calibration-error differences.
        if (
            "absolute_oe_error" in token
            or "absolute_o_e_error" in token
        ):
            digits = (
                5
                if any(
                    term in token
                    for term in [
                        "diff",
                        "difference",
                        "delta",
                        "change",
                        "minus",
                    ]
                )
                else 4
            )
            display[column] = display[column].map(
                lambda x, d=digits: _fmt_number(x, d)
            )
            continue

        # Operating threshold.
        if "threshold" in token and not any(
            term in token
            for term in [
                "distance",
                "below",
                "above",
            ]
        ):
            display[column] = display[column].map(
                lambda x: _fmt_number(x, 5)
            )
            continue

        # Sensitivity/specificity/PPV/NPV/FPR/FNR and their CIs:
        # all stored as fractions and shown as percentages.
        if any(
            metric in token
            for metric in [
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "precision",
                "fpr",
                "fnr",
                "recall",
            ]
        ):
            display[column] = display[column].map(
                lambda x: _fmt_percent_fraction(x, 1)
            )
            continue

        # Prevalence and mortality rates: 2 decimal percentage because event
        # prevalence is low.
        if (
            token == "prevalence"
            or "mortality_prevalence" in token
        ):
            if token.endswith("_percent") or token.endswith("_percentage"):
                display[column] = display[column].map(
                    lambda x: _fmt_percent_already(x, 2)
                )
            else:
                display[column] = display[column].map(
                    lambda x: _fmt_percent_fraction(x, 2)
                )
            continue

        # General percentages already on 0-100 scale.
        if token.endswith("_percent") or token.endswith("_percentage"):
            display[column] = display[column].map(
                lambda x: _fmt_percent_already(x, 1)
            )
            continue

        # Missing-data fraction and percentile fields are stored as fractions.
        if token in {
            "raw_missing_fraction",
            "predicted_risk_percentile",
        }:
            display[column] = display[column].map(
                lambda x: _fmt_percent_fraction(x, 1)
            )
            continue

        # Calibration-bin / error-review probabilities are stored as fractions.
        if token in {
            "mean_predicted_risk",
            "median_predicted_risk",
            "minimum_predicted_risk",
            "maximum_predicted_risk",
            "observed_mortality",
            "observed_ci_lower_95",
            "observed_ci_upper_95",
            "predicted_probability",
            "distance_below_threshold",
            "distance_from_threshold",
        }:
            digits = (
                2
                if token.startswith("observed_")
                else 3
            )
            display[column] = display[column].map(
                lambda x, d=digits: _fmt_probability_percent(x, d)
            )
            continue

        # Odds/hazard ratios and all associated confidence limits.
        if table_token == normalize_name("Table S12a"):
            if (
                token in {
                    "or",
                    "odds_ratio",
                    "hr",
                    "hazard_ratio",
                }
                or "ci" in token
                and any(
                    key in token
                    for key in [
                        "low",
                        "lower",
                        "high",
                        "upper",
                        "95",
                    ]
                )
            ):
                display[column] = display[column].map(
                    lambda x: _fmt_number(x, 2)
                )
                continue

        # SHAP / attribution values: 4 decimals is sufficient for publication
        # while preserving useful ranking information.
        if any(
            word in token
            for word in [
                "attribution",
                "shap",
                "importance",
            ]
        ) and "percent" not in token:
            display[column] = display[column].map(
                lambda x: _fmt_number(x, 4)
            )
            continue

        # Attribution percentages.
        if (
            "percent" in token
            and any(
                word in token
                for word in [
                    "attribution",
                    "importance",
                    "contribution",
                ]
            )
        ):
            display[column] = display[column].map(
                lambda x: _fmt_percent_already(x, 1)
            )
            continue

        # Spearman/Pearson correlations.
        if token in {
            "spearman_rho",
            "rho",
            "pearson_r",
            "correlation",
        }:
            display[column] = display[column].map(
                lambda x: _fmt_number(x, 3)
            )
            continue

        # Clinical continuous variables in audit tables.
        if any(
            term in token
            for term in [
                "age",
                "bmi",
                "duration",
                "blood_loss",
                "fibrinogen",
                "platelet",
                "glucose",
            ]
        ):
            if pd.api.types.is_numeric_dtype(display[column]):
                display[column] = display[column].map(
                    lambda x: _fmt_number(x, 1)
                )
                continue

    return display




# =============================================================================
# COMPACT WORD-TABLE PRESENTATION
# =============================================================================
#
# The CSVs remain complete. These helpers affect only the Word compilation.
# Wide technical tables are reduced to publication-facing columns and, where
# helpful, split into two compact Word tables.

def _value_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text_value = str(value).strip()
    if text_value.lower() in {"nan", "none"}:
        return ""
    return text_value


def _column_by_candidates(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    return first_existing_column(frame, candidates)


def _combine_ci_display(
    frame: pd.DataFrame,
    estimate_candidates: Sequence[str],
    low_candidates: Sequence[str],
    high_candidates: Sequence[str],
    *,
    label: str,
) -> Optional[pd.Series]:
    estimate_col = _column_by_candidates(frame, estimate_candidates)
    if estimate_col is None:
        return None

    low_col = _column_by_candidates(frame, low_candidates)
    high_col = _column_by_candidates(frame, high_candidates)

    values: List[str] = []
    for index in frame.index:
        estimate = _value_text(frame.at[index, estimate_col])

        if not estimate:
            values.append("")
            continue

        low = (
            _value_text(frame.at[index, low_col])
            if low_col is not None
            else ""
        )
        high = (
            _value_text(frame.at[index, high_col])
            if high_col is not None
            else ""
        )

        if low and high:
            values.append(f"{estimate} ({low}-{high})")
        else:
            values.append(estimate)

    return pd.Series(values, index=frame.index, name=label)


def _combine_two_columns(
    frame: pd.DataFrame,
    first_candidates: Sequence[str],
    second_candidates: Sequence[str],
    *,
    separator: str = " / ",
) -> Optional[pd.Series]:
    first_col = _column_by_candidates(frame, first_candidates)
    second_col = _column_by_candidates(frame, second_candidates)

    if first_col is None and second_col is None:
        return None

    values: List[str] = []
    for index in frame.index:
        first = (
            _value_text(frame.at[index, first_col])
            if first_col is not None
            else ""
        )
        second = (
            _value_text(frame.at[index, second_col])
            if second_col is not None
            else ""
        )

        if first and second:
            values.append(f"{first}{separator}{second}")
        else:
            values.append(first or second)

    return pd.Series(values, index=frame.index)


def _select_columns_if_present(
    frame: pd.DataFrame,
    column_specs: Sequence[Tuple[str, Sequence[str]]],
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)

    for label, candidates in column_specs:
        column = _column_by_candidates(frame, candidates)
        if column is not None:
            output[label] = frame[column]

    return output


def _build_s4_publication_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fixed Table S4 Word layout:
      Comparison | Metric | Difference (95% CI) | p-value | Bootstrap replicates

    No provenance paths, analysis labels, removed-feature columns, or empty
    schema-union columns are permitted in the manuscript Word table.
    """
    required = [
        "Reference model",
        "Comparator",
        "Metric",
        "Difference",
        "CI low",
        "CI high",
        "p-value",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Table S4 canonical dataframe is missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(index=frame.index)

    out["Comparison"] = (
        frame["Reference model"].astype(str)
        + " vs "
        + frame["Comparator"].astype(str)
    )

    out["Metric"] = frame["Metric"].astype(str)

    combined: List[str] = []

    for index in frame.index:
        metric = normalize_name(frame.at[index, "Metric"])

        difference = frame.at[index, "Difference"]
        low = frame.at[index, "CI low"]
        high = frame.at[index, "CI high"]

        if "brier" in metric:
            digits = 6
        else:
            digits = 4

        combined.append(
            f"{_fmt_number(difference, digits)} "
            f"({_fmt_number(low, digits)}–{_fmt_number(high, digits)})"
        )

    out["Difference (95% CI)"] = combined
    out["p-value"] = frame["p-value"].map(_fmt_pvalue)

    if "Bootstrap replicates" in frame.columns:
        bootstrap_numeric = pd.to_numeric(
            frame["Bootstrap replicates"],
            errors="coerce",
        )
        if bootstrap_numeric.notna().any():
            out["Bootstrap replicates"] = (
                bootstrap_numeric.map(_fmt_integer)
            )

    return out.reset_index(drop=True)


def _paired_compact_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact paired-comparison tables into:
      comparison/model | metric | difference (95% CI) | p-value
    """
    out = pd.DataFrame(index=frame.index)

    model_a = _column_by_candidates(
        frame,
        [
            "model_a",
            "model_1",
            "reference_model",
            "complete_model",
            "full_model",
            "model",
        ],
    )
    model_b = _column_by_candidates(
        frame,
        [
            "model_b",
            "model_2",
            "comparison_model",
            "removed_model",
            "reduced_model",
        ],
    )

    comparison_col = _column_by_candidates(
        frame,
        [
            "comparison",
            "contrast",
            "model_comparison",
        ],
    )

    if comparison_col is not None:
        out["Comparison"] = frame[comparison_col].map(_value_text)

    elif model_a is not None and model_b is not None:
        out["Comparison"] = [
            (
                f"{_value_text(frame.at[i, model_a])} vs "
                f"{_value_text(frame.at[i, model_b])}"
            )
            for i in frame.index
        ]

    elif model_a is not None:
        out["Model"] = frame[model_a].map(_value_text)

    domain_col = _column_by_candidates(
        frame,
        [
            "removed_domain",
            "domain_removed",
            "domain",
            "feature_group_removed",
        ],
    )
    if domain_col is not None:
        out["Removed domain"] = frame[domain_col].map(_value_text)

    metric_col = _column_by_candidates(
        frame,
        [
            "metric",
            "metric_name",
            "measure",
            "performance_metric",
        ],
    )
    if metric_col is not None:
        out["Metric"] = frame[metric_col].map(_value_text)

    diff_col = _column_by_candidates(
        frame,
        [
            "difference",
            "diff",
            "delta",
            "mean_difference",
            "estimate_difference",
            "complete_minus_removed",
            "full_minus_removed",
        ],
    )
    low_col = _column_by_candidates(
        frame,
        [
            "ci_lower",
            "ci_low",
            "ci_lower_95",
            "lower_95",
            "difference_ci_lower_95",
            "diff_ci_lower_95",
            "delta_ci_lower_95",
        ],
    )
    high_col = _column_by_candidates(
        frame,
        [
            "ci_upper",
            "ci_high",
            "ci_upper_95",
            "upper_95",
            "difference_ci_upper_95",
            "diff_ci_upper_95",
            "delta_ci_upper_95",
        ],
    )

    if diff_col is not None:
        values: List[str] = []
        for i in frame.index:
            diff = _value_text(frame.at[i, diff_col])
            low = (
                _value_text(frame.at[i, low_col])
                if low_col is not None
                else ""
            )
            high = (
                _value_text(frame.at[i, high_col])
                if high_col is not None
                else ""
            )

            if diff and low and high:
                values.append(f"{diff} ({low}-{high})")
            else:
                values.append(diff)

        out["Difference (95% CI)"] = values

    p_col = _column_by_candidates(
        frame,
        [
            "p",
            "p_value",
            "pvalue",
            "paired_p",
            "bootstrap_p",
            "wald_p",
        ],
    )
    if p_col is not None:
        out["p-value"] = frame[p_col].map(_value_text)

    # Retain estimate columns only when no compact difference could be found.
    if "Difference (95% CI)" not in out.columns:
        estimate_a = _column_by_candidates(
            frame,
            ["estimate_a", "value_a", "complete_value", "full_value"],
        )
        estimate_b = _column_by_candidates(
            frame,
            ["estimate_b", "value_b", "removed_value", "reduced_value"],
        )
        if estimate_a is not None:
            out["Complete"] = frame[estimate_a].map(_value_text)
        if estimate_b is not None:
            out["Comparison"] = frame[estimate_b].map(_value_text)

    return out if not out.empty else frame.copy()


def _combine_difference_ci_by_metric(
    frame: pd.DataFrame,
    difference_col: str,
    low_col: str,
    high_col: str,
    metric_col: str,
) -> pd.Series:
    values: List[str] = []

    for index in frame.index:
        metric = normalize_name(frame.at[index, metric_col])

        if "brier" in metric:
            digits = 6
        elif "auroc" in metric or "auprc" in metric:
            digits = 4
        else:
            digits = 4

        values.append(
            f"{_fmt_number(frame.at[index, difference_col], digits)} "
            f"({_fmt_number(frame.at[index, low_col], digits)}–"
            f"{_fmt_number(frame.at[index, high_col], digits)})"
        )

    return pd.Series(values, index=frame.index)


def _build_s5_publication_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compact fitted-model masking table without provenance/path columns."""
    required = [
        "domain",
        "baseline_AUROC",
        "masked_AUROC",
        "AUROC_drop",
        "baseline_AUPRC",
        "masked_AUPRC",
        "AUPRC_drop",
    ]

    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            "Table S5 is missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(
        {
            "Domain": frame["domain"],
            "Baseline AUROC": frame["baseline_AUROC"],
            "Masked AUROC": frame["masked_AUROC"],
            "ΔAUROC": frame["AUROC_drop"],
            "Baseline AUPRC": frame["baseline_AUPRC"],
            "Masked AUPRC": frame["masked_AUPRC"],
            "ΔAUPRC": frame["AUPRC_drop"],
        }
    )

    return out.reset_index(drop=True)


def _build_lodo_paired_publication_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact Table S6b:
      Model | Removed domain | Metric | Full | LODO |
      Difference (95% CI) | p-value | Bootstrap replicates
    """
    required = [
        "model",
        "metric",
        "removed_domain",
        "full_estimate",
        "retrained_estimate",
        "difference_favoring_complete_model",
        "CI_low",
        "CI_high",
        "bootstrap_p_two_sided",
        "n_bootstrap",
    ]

    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            "Table S6b is missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(index=frame.index)
    out["Model"] = frame["model"]
    out["Removed domain"] = frame["removed_domain"]
    out["Metric"] = frame["metric"]
    out["Full model"] = frame["full_estimate"]
    out["Domain-removed model"] = frame["retrained_estimate"]
    out["Difference (95% CI)"] = _combine_difference_ci_by_metric(
        frame,
        "difference_favoring_complete_model",
        "CI_low",
        "CI_high",
        "metric",
    )
    out["p-value"] = frame["bootstrap_p_two_sided"].map(_fmt_pvalue)
    out["Bootstrap replicates"] = frame["n_bootstrap"].map(_fmt_integer)

    return out.reset_index(drop=True)


def _build_eyeear_paired_publication_view(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact Table S12c:
      Model | Metric | Full | Eye/Ear removed |
      Difference (95% CI) | p-value | Bootstrap replicates
    """
    required = [
        "model",
        "metric",
        "full_estimate",
        "retrained_estimate",
        "difference_favoring_complete_model",
        "CI_low",
        "CI_high",
        "bootstrap_p_two_sided",
        "n_bootstrap",
    ]

    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            "Table S12c is missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(index=frame.index)
    out["Model"] = frame["model"]
    out["Metric"] = frame["metric"]
    out["Full model"] = frame["full_estimate"]
    out["Eye/Ear removed"] = frame["retrained_estimate"]
    out["Difference (95% CI)"] = _combine_difference_ci_by_metric(
        frame,
        "difference_favoring_complete_model",
        "CI_low",
        "CI_high",
        "metric",
    )
    out["p-value"] = frame["bootstrap_p_two_sided"].map(_fmt_pvalue)
    out["Bootstrap replicates"] = frame["n_bootstrap"].map(_fmt_integer)

    return out.reset_index(drop=True)


def build_compact_word_views(
    formatted_frame: pd.DataFrame,
    table_id: str,
) -> List[Tuple[Optional[str], pd.DataFrame]]:
    """
    Return one or more compact publication-facing Word views.

    This does NOT alter the CSV. The returned data frames are used only when
    constructing All_Tables_TemporalSafe_StrictOOF.docx.
    """
    table_token = normalize_name(table_id)
    frame = formatted_frame.copy()

    # ------------------------------------------------------------------
    # Table S3 — split model metrics into discrimination/calibration and
    # operating-point performance.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S3"):
        model_col = _column_by_candidates(frame, ["model", "Model"])
        if model_col is None:
            return [(None, frame)]

        s3a = pd.DataFrame({"Model": frame[model_col]})

        auroc = _combine_ci_display(
            frame,
            ["AUROC"],
            ["AUROC_CI_low", "AUROC_ci_lower_95", "AUROC_low"],
            ["AUROC_CI_high", "AUROC_ci_upper_95", "AUROC_high"],
            label="AUROC (95% CI)",
        )
        if auroc is not None:
            s3a[auroc.name] = auroc

        auprc = _combine_ci_display(
            frame,
            ["AUPRC"],
            ["AUPRC_CI_low", "AUPRC_ci_lower_95", "AUPRC_low"],
            ["AUPRC_CI_high", "AUPRC_ci_upper_95", "AUPRC_high"],
            label="AUPRC (95% CI)",
        )
        if auprc is not None:
            s3a[auprc.name] = auprc

        for label, candidates in [
            ("Brier", ["Brier", "brier"]),
            (
                "Calibration slope",
                ["calibration_slope", "Calibration_Slope"],
            ),
            (
                "Calibration intercept",
                ["calibration_intercept", "Calibration_Intercept"],
            ),
            ("ECE", ["ECE", "ece", "expected_calibration_error"]),
        ]:
            col = _column_by_candidates(frame, candidates)
            if col is not None:
                s3a[label] = frame[col]

        s3b = pd.DataFrame({"Model": frame[model_col]})
        for label, candidates in [
            ("Threshold", ["Threshold", "threshold"]),
            ("Sensitivity", ["Sensitivity", "sensitivity", "Recall", "recall"]),
            ("Specificity", ["Specificity", "specificity"]),
            ("PPV", ["PPV", "ppv", "precision_PPV", "precision"]),
            ("Alerts (%)", ["Alerts (%)", "alert_rate", "alert_percent", "alerted_percent"]),
        ]:
            col = _column_by_candidates(frame, candidates)
            if col is not None:
                s3b[label] = frame[col]

        confusion_cols = [
            _column_by_candidates(frame, ["TP", "tp"]),
            _column_by_candidates(frame, ["FP", "fp"]),
            _column_by_candidates(frame, ["FN", "fn"]),
            _column_by_candidates(frame, ["TN", "tn"]),
        ]
        if any(col is not None for col in confusion_cols):
            values = []
            for i in frame.index:
                parts = []
                labels = ["TP", "FP", "FN", "TN"]
                for label, col in zip(labels, confusion_cols):
                    if col is not None:
                        value = _value_text(frame.at[i, col])
                        if value:
                            parts.append(f"{label} {value}")
                values.append("; ".join(parts))
            s3b["Confusion counts"] = values

        return [
            ("Table S3a. Discrimination and calibration", s3a),
            ("Table S3b. Operating-point performance", s3b),
        ]

    # ------------------------------------------------------------------
    # Table S4 — overall paired-bootstrap model comparisons.
    # This table has its own strict canonical layout.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S4"):
        return [(None, _build_s4_publication_view(frame))]

    # ------------------------------------------------------------------
    # Table S6b — leave-one-domain-out paired bootstrap.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S6b"):
        return [(None, _build_lodo_paired_publication_view(frame))]

    # ------------------------------------------------------------------
    # Table S12c — Eye/Ear feature-removal paired bootstrap.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S12c"):
        return [(None, _build_eyeear_paired_publication_view(frame))]

    # ------------------------------------------------------------------
    # Table S5 — compact fitted-model domain masking.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S5"):
        return [(None, _build_s5_publication_view(frame))]

    # ------------------------------------------------------------------
    # Table S6a — keep only the retraining quantities needed to understand
    # complete vs leave-one-domain-out performance.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S6a"):
        out = _select_columns_if_present(
            frame,
            [
                (
                    "Removed domain",
                    [
                        "removed_domain",
                        "domain_removed",
                        "domain",
                        "feature_group_removed",
                    ],
                ),
                ("Model", ["model", "Model"]),
                ("AUROC", ["AUROC", "auroc"]),
                ("AUPRC", ["AUPRC", "auprc"]),
                ("Brier", ["Brier", "brier"]),
            ],
        )
        return [(None, out if not out.empty else frame)]

    # ------------------------------------------------------------------
    # Table S7 — split subgroup table into discrimination and
    # operating-point views.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S7"):
        label_col = _column_by_candidates(
            frame,
            ["display_label", "subgroup_label"],
        )
        if label_col is None:
            label_col = _column_by_candidates(
                frame,
                ["subgroup_level", "subgroup_variable"],
            )

        if label_col is None:
            return [(None, frame)]

        discrimination = pd.DataFrame(
            {"Subgroup": frame[label_col]}
        )

        for label, candidates in [
            ("N", ["n"]),
            ("Events", ["events"]),
            ("Prevalence", ["prevalence"]),
        ]:
            col = _column_by_candidates(frame, candidates)
            if col is not None:
                discrimination[label] = frame[col]

        auroc = _combine_ci_display(
            frame,
            ["AUROC", "auroc"],
            ["AUROC_ci_lower_95", "auroc_ci_lower_95"],
            ["AUROC_ci_upper_95", "auroc_ci_upper_95"],
            label="AUROC (95% CI)",
        )
        if auroc is not None:
            discrimination[auroc.name] = auroc

        auprc = _combine_ci_display(
            frame,
            ["AUPRC", "auprc"],
            ["AUPRC_ci_lower_95", "auprc_ci_lower_95"],
            ["AUPRC_ci_upper_95", "auprc_ci_upper_95"],
            label="AUPRC (95% CI)",
        )
        if auprc is not None:
            discrimination[auprc.name] = auprc

        brier_col = _column_by_candidates(frame, ["Brier", "brier"])
        if brier_col is not None:
            discrimination["Brier"] = frame[brier_col]

        operating = pd.DataFrame(
            {"Subgroup": frame[label_col]}
        )

        metric_specs = [
            (
                "Sensitivity (95% CI)",
                ["sensitivity"],
                ["sensitivity_ci_lower_95"],
                ["sensitivity_ci_upper_95"],
            ),
            (
                "Specificity (95% CI)",
                ["specificity"],
                ["specificity_ci_lower_95"],
                ["specificity_ci_upper_95"],
            ),
            (
                "PPV (95% CI)",
                ["PPV", "ppv"],
                ["PPV_ci_lower_95", "ppv_ci_lower_95"],
                ["PPV_ci_upper_95", "ppv_ci_upper_95"],
            ),
            (
                "O:E ratio (95% CI)",
                ["observed_expected_ratio"],
                ["observed_expected_ratio_ci_lower_95"],
                ["observed_expected_ratio_ci_upper_95"],
            ),
        ]

        for label, estimate, low, high in metric_specs:
            combined = _combine_ci_display(
                frame,
                estimate,
                low,
                high,
                label=label,
            )
            if combined is not None:
                operating[label] = combined

        return [
            (
                "Table S7a. Subgroup discrimination",
                discrimination,
            ),
            (
                "Table S7b. Subgroup operating-point performance",
                operating,
            ),
        ]

    # ------------------------------------------------------------------
    # Table S8 — combine observed CI and predicted-risk range.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S8"):
        out = _select_columns_if_present(
            frame,
            [
                ("Risk bin", ["risk_bin"]),
                ("N", ["n"]),
                ("Events", ["events"]),
                (
                    "Mean predicted risk",
                    ["mean_predicted_risk"],
                ),
            ],
        )

        observed = _combine_ci_display(
            frame,
            ["observed_mortality"],
            ["observed_ci_lower_95"],
            ["observed_ci_upper_95"],
            label="Observed mortality (95% CI)",
        )
        if observed is not None:
            out[observed.name] = observed

        risk_range = _combine_two_columns(
            frame,
            ["minimum_predicted_risk"],
            ["maximum_predicted_risk"],
            separator="-",
        )
        if risk_range is not None:
            out["Predicted-risk range"] = risk_range

        return [(None, out if not out.empty else frame)]

    # ------------------------------------------------------------------
    # Table S9 — already compact; remove repeated provenance columns.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S9"):
        out = _select_columns_if_present(
            frame,
            [
                ("Error group", ["error_group"]),
                ("N", ["n"]),
                (
                    "Age >=70",
                    ["Age_70_or_older_percent"],
                ),
                (
                    "ASA III-V",
                    ["ASA_III_to_V_percent"],
                ),
                (
                    "Emergency",
                    ["Emergency_percent"],
                ),
                (
                    "Median predicted risk",
                    ["median_predicted_risk"],
                ),
            ],
        )
        return [(None, out if not out.empty else frame)]

    # ------------------------------------------------------------------
    # Table S11 — split DARN and XGBoost attribution summaries.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S11"):
        component_col = _column_by_candidates(
            frame,
            ["component"],
        )
        if component_col is None:
            return [(None, frame)]

        views: List[Tuple[Optional[str], pd.DataFrame]] = []

        for component_value in pd.unique(frame[component_col]):
            subset = frame[
                frame[component_col].astype(str)
                == str(component_value)
            ].copy()

            feature_col = _column_by_candidates(
                subset,
                [
                    "pretty_feature",
                    "feature",
                    "feature_name",
                    "transformed_feature",
                ],
            )

            out = pd.DataFrame(index=subset.index)

            rank_col = _column_by_candidates(subset, ["rank"])
            if rank_col is not None:
                out["Rank"] = subset[rank_col]

            if feature_col is not None:
                out["Feature"] = subset[feature_col]

            for label, candidates in [
                (
                    "Mean |attribution|",
                    [
                        "mean_absolute_direct_attribution",
                        "mean_abs_shap",
                        "mean_absolute_shap",
                        "mean_abs_attribution",
                        "mean_absolute_attribution",
                    ],
                ),
                (
                    "Mean signed attribution",
                    [
                        "mean_signed_direct_attribution",
                        "mean_signed_shap",
                        "mean_signed_attribution",
                    ],
                ),
            ]:
                col = _column_by_candidates(subset, candidates)
                if col is not None:
                    out[label] = subset[col]

            if out.empty:
                out = subset.drop(
                    columns=[
                        c
                        for c in [
                            component_col,
                            _column_by_candidates(
                                subset,
                                ["attribution_method"],
                            ),
                        ]
                        if c is not None
                    ],
                    errors="ignore",
                )

            component_text = str(component_value)
            method_col = _column_by_candidates(
                subset,
                ["attribution_method"],
            )
            method_text = (
                str(subset[method_col].iloc[0])
                if method_col is not None and len(subset)
                else ""
            )

            subtitle = (
                f"{component_text} ({method_text})"
                if method_text
                else component_text
            )
            views.append((subtitle, out.reset_index(drop=True)))

        return views

    # ------------------------------------------------------------------
    # Table S12a — adjusted association.
    #
    # The source CSV also contains two descriptive mortality rows with no
    # odds ratio.  They are not regression models and must not become blank
    # rows in the Word table.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S12a"):
        odds_col = _column_by_candidates(
            frame,
            ["odds_ratio", "OR", "or"],
        )
        if odds_col is None:
            raise ValueError("Table S12a has no odds-ratio column.")

        valid = frame[odds_col].map(_nonempty_cell)
        subset = frame.loc[valid].copy()

        out = pd.DataFrame(index=subset.index)

        adjustment_col = _column_by_candidates(
            subset,
            [
                "model_stage",
                "model",
                "adjustment",
                "adjustment_model",
                "specification",
                "covariates",
            ],
        )
        if adjustment_col is None:
            raise ValueError(
                "Table S12a cannot resolve the adjustment/model-stage label."
            )

        out["Adjustment"] = subset[adjustment_col]

        or_series = _combine_ci_display(
            subset,
            ["odds_ratio", "OR", "or"],
            ["CI_low", "ci_lower", "ci_low", "or_ci_low", "OR_CI_low"],
            ["CI_high", "ci_upper", "ci_high", "or_ci_high", "OR_CI_high"],
            label="OR (95% CI)",
        )
        if or_series is None:
            raise ValueError("Table S12a cannot construct OR (95% CI).")
        out[or_series.name] = or_series

        p_col = _column_by_candidates(
            subset,
            ["p_value", "p", "pvalue", "wald_p"],
        )
        if p_col is None:
            raise ValueError("Table S12a has no p-value column.")
        out["p-value"] = subset[p_col]

        return [(None, out.reset_index(drop=True))]

    # ------------------------------------------------------------------
    # Table S12b — full vs Eye/Ear-removed performance.
    # ------------------------------------------------------------------
    if table_token == normalize_name("Table S12b"):
        out = _select_columns_if_present(
            frame,
            [
                ("Model", ["model", "Model"]),
                (
                    "Analysis",
                    [
                        "analysis",
                        "scenario",
                        "feature_set",
                        "condition",
                    ],
                ),
                ("AUROC", ["AUROC", "auroc"]),
                ("AUPRC", ["AUPRC", "auprc"]),
                ("Brier", ["Brier", "brier"]),
            ],
        )
        return [(None, out if not out.empty else frame)]

    # Default: keep the formatted table unchanged.
    return [(None, frame)]




# =============================================================================
# WORD-TABLE SAFETY / BALANCED COLUMN SPLITTING
# =============================================================================
#
# Purpose:
#   Keep the Word document readable WITHOUT accidentally creating sparse or
#   nearly empty compact tables when source-column names differ from expected
#   aliases.
#
# Rules:
#   1. Never write a compact view that has fewer than 2 useful columns.
#   2. Never write a view whose non-label columns are entirely blank.
#   3. If a compact view is not informative, fall back to the original formatted
#      table split into horizontal blocks of at most 8 columns.
#   4. For very wide source tables, horizontal splitting is preferred over tiny
#      fonts. Identifier columns are repeated in each block.
#
# CSV outputs are untouched.

WORD_MAX_COLUMNS = 9

WORD_IDENTIFIER_CANDIDATES = [
    "model",
    "Model",
    "comparison",
    "Comparison",
    "metric",
    "Metric",
    "subgroup",
    "Subgroup",
    "subgroup_label",
    "display_label",
    "removed_domain",
    "Removed domain",
    "domain",
    "Domain",
    "rank",
    "Rank",
    "feature",
    "Feature",
    "component",
    "Component",
    "risk_bin",
    "Risk bin",
    "error_group",
    "Error group",
    "adjustment",
    "Adjustment",
    "analysis",
    "Analysis",
]


def _nonempty_cell(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text_value = str(value).strip()
    return text_value.lower() not in {"", "nan", "none", "na", "n/a"}


def _series_has_data(series: pd.Series) -> bool:
    return bool(series.map(_nonempty_cell).any())


def _frame_has_meaningful_data(frame: pd.DataFrame) -> bool:
    """
    A useful Word table should contain at least two columns and at least one
    populated non-identifier/data column.
    """
    if frame is None or frame.empty or frame.shape[1] < 2:
        return False

    identifier_tokens = {
        _normalize_column_token(candidate)
        for candidate in WORD_IDENTIFIER_CANDIDATES
    }

    data_columns = [
        column
        for column in frame.columns
        if _normalize_column_token(column) not in identifier_tokens
    ]

    # If every column is an identifier/label column (e.g. Model + Metric),
    # the compact view is NOT useful. Force fallback to the complete source
    # table rather than silently emitting a table with no estimates/CI/p-values.
    if not data_columns:
        return False

    return any(
        _series_has_data(frame[column])
        for column in data_columns
    )


def _detect_identifier_columns(
    frame: pd.DataFrame,
    *,
    max_identifiers: int = 2,
) -> List[str]:
    """
    Choose up to two useful label columns to repeat across horizontally split
    Word tables.
    """
    identifiers: List[str] = []

    # Exact/case-insensitive candidate search first.
    normalized_map = {
        _normalize_column_token(column): column
        for column in frame.columns
    }

    for candidate in WORD_IDENTIFIER_CANDIDATES:
        token = _normalize_column_token(candidate)
        column = normalized_map.get(token)
        if column is not None and column not in identifiers:
            identifiers.append(column)
            if len(identifiers) >= max_identifiers:
                break

    # If no known identifier exists, repeat first non-numeric/text-like column.
    if not identifiers:
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                identifiers.append(column)
                break

    # Final fallback: first column.
    if not identifiers and len(frame.columns):
        identifiers.append(frame.columns[0])

    return identifiers[:max_identifiers]


def split_wide_word_table(
    frame: pd.DataFrame,
    *,
    max_columns: int = WORD_MAX_COLUMNS,
    title_prefix: Optional[str] = None,
) -> List[Tuple[Optional[str], pd.DataFrame]]:
    """
    Split a wide table into readable horizontal blocks while repeating one or
    two identifier columns. No source columns are discarded.
    """
    if frame is None or frame.empty:
        return [(title_prefix, frame)]

    if frame.shape[1] <= max_columns:
        return [(title_prefix, frame)]

    identifiers = _detect_identifier_columns(frame)

    metric_columns = [
        column
        for column in frame.columns
        if column not in identifiers
    ]

    slots = max(1, max_columns - len(identifiers))

    views: List[Tuple[Optional[str], pd.DataFrame]] = []
    total_parts = (len(metric_columns) + slots - 1) // slots

    for part_index, start in enumerate(
        range(0, len(metric_columns), slots),
        start=1,
    ):
        chunk = metric_columns[start:start + slots]
        selected = identifiers + chunk

        subtitle = title_prefix
        if total_parts > 1:
            part_label = f"Part {part_index} of {total_parts}"
            subtitle = (
                f"{title_prefix} — {part_label}"
                if title_prefix
                else part_label
            )

        views.append(
            (
                subtitle,
                frame[selected].copy(),
            )
        )

    return views


def _paired_view_has_statistics(
    frame: pd.DataFrame,
) -> bool:
    """
    Paired-comparison Word views must contain at least one statistical result
    column beyond label columns such as Model / Removed domain / Metric.
    """
    if frame is None or frame.empty:
        return False

    tokens = {
        _normalize_column_token(column)
        for column in frame.columns
    }

    # Preferred compact outputs.
    if any(
        token in tokens
        for token in {
            "difference_95_ci",
            "difference",
            "diff",
            "delta",
            "p_value",
            "pvalue",
            "complete",
            "comparison",
        }
    ):
        # "comparison" by itself can be a label, so require another
        # statistical-looking column unless Difference is present.
        statistical = [
            token
            for token in tokens
            if any(
                key in token
                for key in [
                    "difference",
                    "diff",
                    "delta",
                    "ci",
                    "p_value",
                    "pvalue",
                    "estimate",
                    "complete_value",
                    "removed_value",
                    "full_value",
                ]
            )
        ]
        if statistical:
            return True

    # Generic numeric/statistical columns from the raw paired artifact.
    label_tokens = {
        "model",
        "model_a",
        "model_b",
        "comparison",
        "removed_domain",
        "domain",
        "metric",
        "metric_name",
        "measure",
        "source_file",
        "analysis_method",
    }

    result_columns = [
        column
        for column in frame.columns
        if _normalize_column_token(column) not in label_tokens
        and _series_has_data(frame[column])
    ]

    return len(result_columns) > 0


def build_safe_word_views(
    formatted_frame: pd.DataFrame,
    table_id: str,
) -> List[Tuple[Optional[str], pd.DataFrame]]:
    """
    Use the compact publication view when it is informative. If the compact
    logic produces a sparse/empty result because column aliases do not match,
    automatically fall back to the complete formatted source table split into
    readable horizontal blocks.

    Also split any still-wide compact view into <=8-column blocks.
    """
    compact_views = build_compact_word_views(
        formatted_frame,
        table_id,
    )

    safe_views: List[Tuple[Optional[str], pd.DataFrame]] = []

    paired_table = normalize_name(table_id) in {
        normalize_name("Table S6b"),
        normalize_name("Table S12c"),
    }

    for subtitle, view in compact_views:
        paired_invalid = (
            paired_table
            and not _paired_view_has_statistics(view)
        )

        if paired_invalid or not _frame_has_meaningful_data(view):
            # Alias mismatch or over-aggressive reduction: preserve all source
            # data by falling back to a horizontally split full table.
            if paired_invalid:
                fallback_title = (
                    f"{subtitle} — complete paired-bootstrap results"
                    if subtitle
                    else "Complete paired-bootstrap results"
                )
            else:
                fallback_title = (
                    f"{subtitle} — complete data"
                    if subtitle
                    else None
                )
            fallback_views = split_wide_word_table(
                formatted_frame,
                max_columns=WORD_MAX_COLUMNS,
                title_prefix=fallback_title,
            )
            safe_views.extend(fallback_views)
            continue

        # Compact view is informative. If it is still too wide, split it rather
        # than shrinking text.
        split_views = split_wide_word_table(
            view,
            max_columns=WORD_MAX_COLUMNS,
            title_prefix=subtitle,
        )
        safe_views.extend(split_views)

    # Ultimate safety net.
    if not safe_views:
        safe_views = split_wide_word_table(
            formatted_frame,
            max_columns=WORD_MAX_COLUMNS,
            title_prefix=None,
        )

    return safe_views




# =============================================================================
# WORD COMPILATION
# =============================================================================

def set_landscape(section) -> None:
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches

    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)



def _format_ci_string(
    value: Any,
    *,
    kind: str,
) -> str:
    """
    Normalize combined 'estimate (low-high)' strings after compact-table
    construction. Handles the main display types used in the supplement.
    """
    text_value = _value_text(value)
    if not text_value:
        return ""

    # If no parenthesized interval is present, preserve the already-formatted
    # value.
    match = re.match(
        r"^\s*([^\(]+?)\s*\(\s*([+-]?[0-9.eE]+)(%)?\s*[-–]\s*([+-]?[0-9.eE]+)(%)?\s*\)\s*$",
        text_value,
    )
    if not match:
        return text_value

    (
        estimate_text,
        low_text,
        low_percent_mark,
        high_text,
        high_percent_mark,
    ) = match.groups()

    try:
        low = float(low_text)
        high = float(high_text)
    except Exception:
        return text_value

    if kind == "percent":
        est = estimate_text.strip()

        if "%" not in est:
            try:
                est = _fmt_percent_fraction(float(est), 1)
            except Exception:
                pass

        # CI values may already have been converted to percentages before the
        # combined string was created. Do not multiply them by 100 twice.
        if low_percent_mark:
            low_fmt = _fmt_percent_already(low, 1)
        else:
            low_fmt = _fmt_percent_fraction(low, 1)

        if high_percent_mark:
            high_fmt = _fmt_percent_already(high, 1)
        else:
            high_fmt = _fmt_percent_fraction(high, 1)

        return f"{est} ({low_fmt}–{high_fmt})"

    if kind == "or":
        try:
            est = _fmt_number(float(estimate_text.strip().replace("%", "")), 2)
        except Exception:
            est = estimate_text.strip()
        return (
            f"{est} "
            f"({_fmt_number(low, 2)}–{_fmt_number(high, 2)})"
        )

    if kind == "metric3":
        try:
            est = _fmt_number(float(estimate_text.strip()), 3)
        except Exception:
            est = estimate_text.strip()
        return (
            f"{est} "
            f"({_fmt_number(low, 3)}–{_fmt_number(high, 3)})"
        )

    if kind == "metric4":
        try:
            est = _fmt_number(float(estimate_text.strip()), 4)
        except Exception:
            est = estimate_text.strip()
        return (
            f"{est} "
            f"({_fmt_number(low, 4)}–{_fmt_number(high, 4)})"
        )

    if kind == "oe":
        try:
            est = _fmt_number(float(estimate_text.strip()), 3)
        except Exception:
            est = estimate_text.strip()
        return (
            f"{est} "
            f"({_fmt_number(low, 3)}–{_fmt_number(high, 3)})"
        )

    return text_value


def finalize_word_completeness(
    frame: pd.DataFrame,
    table_id: str,
) -> pd.DataFrame:
    """
    Remove structural blank rows/columns and ensure that the Word package does
    not silently contain unexplained empty cells.

    Table S10 is a patient-level deidentified error review; genuinely unavailable
    raw measurements are displayed as "NA" rather than blank.
    """
    out = frame.copy()

    def blank(value: Any) -> bool:
        return not _nonempty_cell(value)

    # Drop rows that contain no display information at all.
    row_blank = out.apply(
        lambda row: all(blank(value) for value in row.tolist()),
        axis=1,
    )
    out = out.loc[~row_blank].copy()

    # Drop columns that contain no display information at all.
    drop_columns: List[str] = []
    for column in out.columns:
        if all(blank(value) for value in out[column].tolist()):
            drop_columns.append(column)

    if drop_columns:
        out = out.drop(columns=drop_columns)

    # True clinical missingness in S10 is not a layout error.
    if normalize_name(table_id) == normalize_name("Table S10"):
        out = out.map(
            lambda value: "NA" if blank(value) else value
        )

    # Every other manuscript-facing table should now be complete.
    remaining: List[Tuple[int, str]] = []
    for row_position, (_, row) in enumerate(out.iterrows(), start=1):
        for column in out.columns:
            if blank(row[column]):
                remaining.append((row_position, str(column)))

    if remaining:
        preview = ", ".join(
            f"row {row}, column '{column}'"
            for row, column in remaining[:12]
        )
        raise ValueError(
            f"{table_id} still contains unexpected blank Word cells: "
            f"{preview}"
        )

    return out.reset_index(drop=True)


def finalize_word_decimal_display(
    frame: pd.DataFrame,
    table_id: str,
) -> pd.DataFrame:
    """
    Final display-only sanitation after compact/split Word views are created.
    This catches decimal leakage inside combined CI strings.
    """
    out = frame.copy()
    table_token = normalize_name(table_id)

    for column in out.columns:
        token = _normalize_column_token(column)

        # Combined discrimination intervals.
        if (
            "auroc" in token
            or "auprc" in token
        ) and "ci" in token:
            out[column] = out[column].map(
                lambda x: _format_ci_string(
                    x,
                    kind="metric3",
                )
            )
            continue

        # Combined operating-point intervals.
        if any(
            metric in token
            for metric in [
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "precision",
                "recall",
            ]
        ) and "ci" in token:
            out[column] = out[column].map(
                lambda x: _format_ci_string(
                    x,
                    kind="percent",
                )
            )
            continue

        # O:E ratio intervals.
        if (
            "o_e" in token
            or "oe_ratio" in token
            or "observed_expected" in token
        ) and "ci" in token:
            out[column] = out[column].map(
                lambda x: _format_ci_string(
                    x,
                    kind="oe",
                )
            )
            continue

        # Odds ratio intervals.
        if table_token == normalize_name("Table S12a") and (
            token.startswith("or")
            or "odds_ratio" in token
        ) and "ci" in token:
            out[column] = out[column].map(
                lambda x: _format_ci_string(
                    x,
                    kind="or",
                )
            )
            continue

        # Paired Difference (95% CI) strings should already have their numeric
        # components formatted by the row-metric policy. Preserve them here.
        if "difference" in token and "ci" in token:
            continue

        # Any remaining attribution/SHAP numeric columns.
        if any(
            key in token
            for key in [
                "attribution",
                "shap",
                "importance",
            ]
        ) and "percent" not in token:
            out[column] = out[column].map(
                lambda x: (
                    _fmt_number(x, 4)
                    if _value_text(x)
                    else ""
                )
            )
            continue

    return out


def dataframe_to_word_table(
    document,
    frame: pd.DataFrame,
    *,
    max_rows: Optional[int] = None,
) -> None:
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    shown = frame.copy()

    if max_rows is not None and len(shown) > max_rows:
        shown = shown.head(max_rows).copy()

    shown = shown.replace({np.nan: ""})

    table = document.add_table(
        rows=1,
        cols=max(1, len(shown.columns)),
    )
    table.style = "Table Grid"
    table.autofit = True

    # Repeat the header row on subsequent pages.
    header_tr = table.rows[0]._tr
    tr_pr = header_tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)

    header = table.rows[0].cells
    for index, column in enumerate(shown.columns):
        header[index].text = str(column)

    for _, row in shown.iterrows():
        cells = table.add_row().cells
        for index, value in enumerate(row.tolist()):
            cells[index].text = str(value)

    # The compact views are designed to stay readable without microscopic text.
    n_columns = len(shown.columns)
    if n_columns <= 6:
        font_size = 9.0
    elif n_columns <= 8:
        font_size = 8.5
    elif n_columns <= 10:
        font_size = 8.0
    else:
        font_size = 7.0

    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

            # Small internal cell margins improve readability.
            tc = cell._tc
            tc_pr = tc.get_or_add_tcPr()
            tc_mar = tc_pr.first_child_found_in("w:tcMar")
            if tc_mar is None:
                tc_mar = OxmlElement("w:tcMar")
                tc_pr.append(tc_mar)

            for margin_name in ["top", "left", "bottom", "right"]:
                node = tc_mar.find(qn(f"w:{margin_name}"))
                if node is None:
                    node = OxmlElement(f"w:{margin_name}")
                    tc_mar.append(node)
                node.set(qn("w:w"), "55")
                node.set(qn("w:type"), "dxa")

            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0

                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font_size)
                    if row_index == 0:
                        run.font.bold = True

    if max_rows is not None and len(frame) > max_rows:
        document.add_paragraph(
            f"Word preview truncated to the first {max_rows} rows. "
            "The accompanying CSV contains the complete table."
        )


def build_tables_word(
    manifest_frame: pd.DataFrame,
    output_path: pathlib.Path,
    run_dir: pathlib.Path,
) -> pathlib.Path:
    # Always enforce the fixed manuscript sequence at rendering time,
    # independent of generation order.
    manifest_frame = sort_publication_manifest(
        manifest_frame
    )

    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for Word compilation."
        ) from exc

    document = Document()
    set_landscape(document.sections[0])

    title = document.add_heading(
        "Publication Tables — Temporal-Safe Strict-OOF Analysis",
        level=0,
    )
    title.runs[0].font.size = Pt(18)

    document.add_paragraph(f"Run: {run_dir}")
    document.add_paragraph(
        "Display values in this Word file follow the locked manuscript rounding "
        "policy. Wide technical tables are shown in compact publication-facing "
        "views. If a compact view cannot safely resolve the source columns, the "
        "complete table is automatically split into readable horizontal parts "
        "rather than dropping data. Machine-readable CSV files retain all "
        "original columns and full numerical precision."
    )

    for row in manifest_frame.itertuples(index=False):
        if getattr(row, "status") != "created":
            continue

        path_text = str(getattr(row, "file"))
        if not path_text:
            continue

        path = pathlib.Path(path_text)
        if not path.exists() or path.suffix.lower() != ".csv":
            continue

        document.add_page_break()
        document.add_heading(
            f"{getattr(row, 'table_id')}. {getattr(row, 'title')}",
            level=1,
        )
        document.add_paragraph(
            f"Source: {getattr(row, 'source')}"
        )

        frame = pd.read_csv(path)

        # Apply the locked publication decimal policy ONLY to the Word display.
        # The underlying CSV remains full precision.
        table_id = str(getattr(row, "table_id"))
        display_frame = format_publication_dataframe(
            frame,
            table_id,
        )

        # Convert wide technical tables into compact publication-facing Word
        # views. This never changes the CSV or analysis results.
        word_views = build_safe_word_views(
            display_frame,
            table_id,
        )

        for view_index, (subtitle, word_frame) in enumerate(word_views):
            if subtitle:
                if view_index > 0:
                    document.add_paragraph()
                document.add_heading(
                    subtitle,
                    level=2,
                )

            # Final display-only decimal sanitation, including combined CI
            # strings created during compact Word-table construction.
            word_frame = finalize_word_decimal_display(
                word_frame,
                table_id,
            )
            word_frame = finalize_word_completeness(
                word_frame,
                table_id,
            )

            # Very long tables remain complete in CSV; Word receives a practical
            # preview so the document stays usable.
            max_rows = 80 if len(word_frame) > 80 else None
            dataframe_to_word_table(
                document,
                word_frame,
                max_rows=max_rows,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return output_path


def figure_sort_key(path: pathlib.Path) -> Tuple[int, int, str]:
    name = path.stem.lower()

    main_match = re.search(
        r"(?:^|[_\s-])(?:fig|figure)[_\s-]?([1-9][0-9]*)(?:[_\s-]|$)",
        name,
    )
    if main_match and not re.search(
        r"(?:figs|figures|supplement)",
        name,
    ):
        return (0, int(main_match.group(1)), name)

    supp_match = re.search(
        r"(?:figs|figures|supplementary[_\s-]?figure)[_\s-]?([0-9]+)",
        name,
    )
    if supp_match:
        return (1, int(supp_match.group(1)), name)

    if "supp" in name or "figs" in name:
        return (1, 999, name)

    return (2, 999, name)


def collect_current_run_figures(
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> List[pathlib.Path]:
    candidates: List[pathlib.Path] = []

    for path in run_dir.rglob("*.png"):
        if is_generated_publication_artifact(path):
            continue

        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass

        candidates.append(path)

    unique = {
        str(path.resolve()): path
        for path in candidates
        if path.exists()
    }

    return sorted(
        unique.values(),
        key=figure_sort_key,
    )


def manuscript_figure_key(path: pathlib.Path) -> Optional[str]:
    """Return manuscript figure ID for exact final-submission basenames."""
    for figure_id, basename in EXPLICIT_RUN_FIGURES.items():
        if path.name == basename:
            return figure_id
    return None


def find_exact_basename(
    root: pathlib.Path,
    basename: str,
    *,
    excluded_root: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Find exactly one file with the requested basename."""
    matches: List[pathlib.Path] = []

    for path in root.rglob(basename):
        if excluded_root is not None:
            try:
                path.relative_to(excluded_root)
                continue
            except ValueError:
                pass
        matches.append(path)

    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Required submission figure not found: {basename}\n"
            f"Searched under: {root}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous submission figure basename '{basename}'. "
            f"Found {len(matches)} files:\n"
            + "\n".join(f"  - {path}" for path in matches)
        )

    return matches[0]


def resolve_submission_figure_set(
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    figure1_path: Optional[pathlib.Path],
    figure2_path: Optional[pathlib.Path],
) -> Tuple[List[pathlib.Path], pd.DataFrame]:
    """
    Resolve the exact 12-figure journal-submission set.

    Figure 1 and Figure 2 must be supplied explicitly if they are not already
    copied into the final run directory. All other figures are resolved by
    exact basename. Missing or ambiguous figures stop the workflow.
    """
    resolved: Dict[str, pathlib.Path] = {}
    rows: List[Dict[str, Any]] = []

    # Figure 1 / Figure 2: explicit CLI path has priority.
    supplied = {
        "Figure 1": figure1_path,
        "Figure 2": figure2_path,
    }

    # Canonical fallback basenames if the user copies these figures into run_dir.
    fallback_names = {
        "Figure 1": "Figure1_Final.png",
        "Figure 2": "Figure2_TemporalSafe_StrictOOF_Cohort_Characteristics.png",
    }

    for figure_id in ["Figure 1", "Figure 2"]:
        requested = supplied[figure_id]

        if requested is not None:
            requested = pathlib.Path(requested)
            if not requested.exists():
                raise FileNotFoundError(
                    f"{figure_id} path does not exist:\n{requested}"
                )
            resolved[figure_id] = requested
            source_mode = "explicit CLI path"
        else:
            # Only accept the canonical basename from the final run. We do not
            # silently search the whole project and risk selecting an old figure.
            fallback = run_dir / fallback_names[figure_id]
            if not fallback.exists():
                raise FileNotFoundError(
                    f"{figure_id} is required for the submission package but is "
                    f"not present in the final run.\n\n"
                    f"Either provide:\n"
                    f"  --{figure_id.lower().replace(' ', '')}_path /path/to/file.png\n"
                    f"or copy the final figure to:\n"
                    f"  {fallback}\n"
                )
            resolved[figure_id] = fallback
            source_mode = "canonical final-run basename"

        rows.append(
            {
                "figure_id": figure_id,
                "include": 1,
                "file": str(resolved[figure_id]),
                "basename": resolved[figure_id].name,
                "source_mode": source_mode,
                "status": "resolved",
            }
        )

    # Figures 3–6 and S1–S6: exact basename only.
    for figure_id, basename in EXPLICIT_RUN_FIGURES.items():
        path = find_exact_basename(
            run_dir,
            basename,
            excluded_root=output_dir,
        )
        resolved[figure_id] = path

        rows.append(
            {
                "figure_id": figure_id,
                "include": 1,
                "file": str(path),
                "basename": path.name,
                "source_mode": "exact final-run basename",
                "status": "resolved",
            }
        )

    ordered_paths = [
        resolved[figure_id]
        for figure_id in SUBMISSION_FIGURE_ORDER
    ]

    manifest = pd.DataFrame(rows)

    order_map = {
        figure_id: index
        for index, figure_id in enumerate(SUBMISSION_FIGURE_ORDER)
    }
    manifest["_order"] = manifest["figure_id"].map(order_map)
    manifest = (
        manifest.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    if len(ordered_paths) != 12:
        raise RuntimeError(
            f"Expected exactly 12 submission figures; resolved {len(ordered_paths)}."
        )

    if len({str(path.resolve()) for path in ordered_paths}) != 12:
        raise RuntimeError(
            "The submission figure set contains duplicate file paths."
        )

    return ordered_paths, manifest



def add_picture_fit_page(
    document,
    image_path: pathlib.Path,
) -> None:
    from docx.shared import Inches

    max_width = 9.75
    max_height = 6.35

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width_px, height_px = image.size

        if width_px <= 0 or height_px <= 0:
            document.add_picture(
                str(image_path),
                width=Inches(max_width),
            )
            return

        aspect = width_px / height_px

        width = max_width
        height = width / aspect

        if height > max_height:
            height = max_height
            width = height * aspect

        document.add_picture(
            str(image_path),
            width=Inches(width),
            height=Inches(height),
        )
    except Exception:
        document.add_picture(
            str(image_path),
            width=Inches(max_width),
        )


def build_figures_word(
    figure_paths: Sequence[pathlib.Path],
    output_path: pathlib.Path,
    run_dir: pathlib.Path,
    *,
    document_title: str,
) -> pathlib.Path:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for Word compilation."
        ) from exc

    document = Document()
    set_landscape(document.sections[0])

    title = document.add_heading(
        document_title,
        level=0,
    )
    title.runs[0].font.size = Pt(18)
    document.add_paragraph(f"Run: {run_dir}")
    document.add_paragraph(
        f"Figures included: {len(figure_paths)}"
    )

    for index, path in enumerate(figure_paths, start=1):
        if index > 1:
            document.add_page_break()

        # Submission documents are passed in manuscript order. For the exact
        # 12-figure set, use that order directly so externally supplied
        # Figure 1 / Figure 2 are labeled correctly.
        if len(figure_paths) == len(SUBMISSION_FIGURE_ORDER):
            heading_text = SUBMISSION_FIGURE_ORDER[index - 1]
        else:
            figure_id = manuscript_figure_key(path)
            heading_text = (
                figure_id
                if figure_id is not None
                else path.stem.replace("_", " ")
            )

        heading = document.add_heading(
            heading_text,
            level=1,
        )
        if heading.runs:
            heading.runs[0].font.size = Pt(13)

        add_picture_fit_page(document, path)

        paragraph = document.add_paragraph(
            f"Source file: {path}"
        )
        for run in paragraph.runs:
            run.font.size = Pt(8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return output_path



# =============================================================================
# MAIN TABLE-BUILDING WORKFLOW
# =============================================================================

def run_all(
    run_dir: pathlib.Path,
    pipeline_module: pathlib.Path,
    data_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    audit_model: str,
    n_bootstrap: int,
    n_jobs: int,
    n_calibration_bins: int,
    top_explainability_features: int,
    skip_tables_word: bool,
    skip_figures_word: bool,
    figure1_path: Optional[pathlib.Path],
    figure2_path: Optional[pathlib.Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []

    print("=" * 100)
    print("BUILDING ALL TABLES FROM FINAL TEMPORAL-SAFE STRICT-OOF RUN")
    print("=" * 100)
    print("Run directory :", run_dir)
    print("Pipeline      :", pipeline_module)
    print("Data          :", data_path)
    print("Output        :", output_dir)

    reconstructed = reconstruct_current_run(
        run_dir,
        pipeline_module,
        data_path,
    )

    # -------------------------------------------------------------------------
    # Table 1
    # -------------------------------------------------------------------------
    table1 = build_table1(reconstructed)
    table1_path = save_table(
        table1,
        output_dir / "Table1_Cohort_Characteristics.csv",
    )
    add_manifest_entry(
        manifest,
        "Table 1",
        "Cohort characteristics stratified by 30-day mortality",
        table1_path,
        "Temporal-safe modeled cohort reconstructed from the final run",
    )

    # -------------------------------------------------------------------------
    # Table 2 + Table S3
    # -------------------------------------------------------------------------
    metrics, metrics_source = load_test_metrics_with_bootstrap_ci(
        run_dir,
        test_n=len(reconstructed["y_test"]),
    )

    table2 = build_main_performance_table(
        metrics,
        test_n=len(reconstructed["y_test"]),
    )
    table2_path = save_table(
        table2,
        output_dir / "Table2_HeldOut_Test_Model_Performance.csv",
    )
    add_manifest_entry(
        manifest,
        "Table 2",
        "Held-out test performance of all saved models",
        table2_path,
        metrics_source,
    )

    table_s3_path = save_table(
        metrics,
        output_dir / "TableS3_Full_HeldOut_Test_Model_Metrics.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S3",
        "Full saved held-out test model metrics",
        table_s3_path,
        metrics_source,
    )

    # -------------------------------------------------------------------------
    # Table S1a / S1b
    # -------------------------------------------------------------------------
    cohort_flow = reconstructed["cohort_flow"].copy()

    for column in [
        "Excluded_at_step",
        "Excluded_events_at_step",
    ]:
        if column in cohort_flow.columns:
            cohort_flow[column] = (
                pd.to_numeric(
                    cohort_flow[column],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

    cohort_flow_path = save_table(
        cohort_flow,
        output_dir / "TableS1a_Cohort_Flow.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S1a",
        "Cohort-flow audit",
        cohort_flow_path,
        "pipeline.apply_reviewer_cohort_exclusions on temporal-safe dataset",
    )

    split_audit_path = save_table(
        reconstructed["split_audit"],
        output_dir / "TableS1b_Development_Test_Split_Audit.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S1b",
        "Development and held-out test split audit",
        split_audit_path,
        "Saved development OOF and test prediction row positions",
    )

    # -------------------------------------------------------------------------
    # Table S2a / S2b
    # -------------------------------------------------------------------------
    fold_summary, domain_summary = build_feature_selection_tables(
        reconstructed,
        run_dir,
    )

    if not fold_summary.empty:
        path = save_table(
            fold_summary,
            output_dir / "TableS2a_Fold_Feature_Selection_Summary.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S2a",
            "Fold-level preprocessing and feature-selection summary",
            path,
            "Saved fold_artifacts from the strict-OOF run",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S2a",
            "Fold-level preprocessing and feature-selection summary",
            None,
            "fold_artifacts",
            status="not found",
        )

    if not domain_summary.empty:
        path = save_table(
            domain_summary,
            output_dir / "TableS2b_Selected_Features_by_Domain_and_Fold.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S2b",
            "Selected transformed features by clinical domain and fold",
            path,
            "selected_domain_metadata.json from each outer fold",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S2b",
            "Selected transformed features by clinical domain and fold",
            None,
            "selected_domain_metadata.json",
            status="not found",
        )

    # -------------------------------------------------------------------------
    # Table S4 — saved paired-bootstrap model comparisons
    # -------------------------------------------------------------------------
    paired = read_paired_model_comparisons(run_dir)

    if paired is not None and not paired.empty:
        path = save_table(
            paired,
            output_dir / "TableS4_Paired_Bootstrap_Model_Comparisons.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S4",
            "Paired-bootstrap model comparisons",
            path,
            "Saved paired-bootstrap comparison artifacts from final run",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S4",
            "Paired-bootstrap model comparisons",
            None,
            "No matching saved paired-bootstrap model-comparison CSV discovered",
            status="not found",
        )

    # -------------------------------------------------------------------------
    # Table S5 — fitted-model input-domain masking
    # -------------------------------------------------------------------------
    masking = read_best_masking_table(run_dir)

    if masking is not None and not masking.empty:
        path = save_table(
            masking,
            output_dir / "TableS5_Uniform_DARN_Input_Domain_Masking.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S5",
            "Uniform DARN fitted-model input-domain masking",
            path,
            "Saved strict-OOF masking artifact",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S5",
            "Uniform DARN fitted-model input-domain masking",
            None,
            "No masking artifact discovered",
            status="not found",
        )

    # -------------------------------------------------------------------------
    # Table S6a / S6b — LODO
    # -------------------------------------------------------------------------
    lodo_metrics, lodo_paired = read_lodo_tables(run_dir)

    if lodo_metrics is not None and not lodo_metrics.empty:
        path = save_table(
            lodo_metrics,
            output_dir / "TableS6a_Leave_One_Domain_Out_Retrained_Test_Metrics.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S6a",
            "Leave-one-domain-out retrained held-out test metrics",
            path,
            "leave_one_domain_out_retraining",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S6a",
            "Leave-one-domain-out retrained held-out test metrics",
            None,
            "leave_one_domain_out_retraining",
            status="not found",
        )

    if lodo_paired is not None and not lodo_paired.empty:
        path = save_table(
            lodo_paired,
            output_dir / "TableS6b_Leave_One_Domain_Out_Paired_Bootstrap.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S6b",
            "Leave-one-domain-out paired-bootstrap differences",
            path,
            "leave_one_domain_out_retraining",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S6b",
            "Leave-one-domain-out paired-bootstrap differences",
            None,
            "leave_one_domain_out_retraining",
            status="not found",
        )

    # -------------------------------------------------------------------------
    # Audit-model-derived Tables S7-S10
    # -------------------------------------------------------------------------
    audit = resolve_audit_model(
        run_dir,
        reconstructed["test_predictions"],
        audit_model,
    )

    y_test = reconstructed["y_test"]
    p_test = audit["probability"]
    threshold = float(audit["threshold"])

    groups = derive_subgroups(reconstructed["test_cohort"])

    subgroup_table = build_subgroup_table(
        y_test,
        p_test,
        threshold,
        groups,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
        random_state=DEFAULT_RANDOM_STATE,
    )
    subgroup_table.insert(0, "audit_model", audit["model_name"])
    subgroup_table.insert(1, "operating_threshold", threshold)

    path = save_table(
        subgroup_table,
        output_dir / "TableS7_Subgroup_Performance_and_Operating_Point_Audit.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S7",
        "Exploratory subgroup performance and operating-point audit",
        path,
        (
            f"Computed from saved test predictions for {audit['model_name']} "
            "and the temporal-safe held-out test cohort"
        ),
    )

    tail_calibration = build_tail_calibration_table(
        y_test,
        p_test,
        n_calibration_bins,
    )
    tail_calibration.insert(0, "audit_model", audit["model_name"])

    path = save_table(
        tail_calibration,
        output_dir / "TableS8_Tail_Calibration_Quantile_Bins.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S8",
        "Tail-focused calibration quantile-bin statistics",
        path,
        "Computed from saved calibrated test probabilities",
    )

    error_groups = assign_error_groups(
        y_test,
        p_test,
        threshold,
    )

    error_profile = build_error_profile(
        reconstructed["test_cohort"],
        error_groups,
        p_test,
        groups,
    )
    error_profile.insert(0, "audit_model", audit["model_name"])
    error_profile.insert(1, "operating_threshold", threshold)

    path = save_table(
        error_profile,
        output_dir / "TableS9_Error_Group_Clinical_Profiles.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S9",
        "TP/FP/FN/TN descriptive clinical profiles",
        path,
        "Computed from saved test probabilities at development-defined threshold",
    )

    false_negative_review = build_false_negative_review(
        reconstructed["test_cohort"],
        p_test,
        threshold,
        error_groups,
        groups,
    )
    if not false_negative_review.empty:
        false_negative_review.insert(
            0,
            "audit_model",
            audit["model_name"],
        )

    path = save_table(
        false_negative_review,
        output_dir / "TableS10_Deidentified_False_Negative_Review.csv",
    )
    add_manifest_entry(
        manifest,
        "Table S10",
        "Deidentified false-negative review",
        path,
        "Computed from held-out test errors; no direct identifiers exported",
    )

    # -------------------------------------------------------------------------
    # Table S11 — explainability outputs from Figure 4
    # -------------------------------------------------------------------------
    explainability = read_explainability_table(
        run_dir,
        top_explainability_features,
    )

    if explainability is not None and not explainability.empty:
        path = save_table(
            explainability,
            output_dir / "TableS11_Component_Explainability_Features.csv",
        )
        add_manifest_entry(
            manifest,
            "Table S11",
            (
                f"Top {top_explainability_features} Uniform DARN GradientSHAP "
                "and XGBoost TreeSHAP features"
            ),
            path,
            "Strict-OOF Figure 4 attribution outputs",
        )
    else:
        add_manifest_entry(
            manifest,
            "Table S11",
            "Component-level explainability features",
            None,
            (
                "Strict-OOF Figure 4 attribution outputs not found. "
                "Run Figure 4 first."
            ),
            status="not found",
        )

    # -------------------------------------------------------------------------
    # Table S12 — manuscript-facing Eye/Ear sensitivity tables only.
    # Row-level probabilities/predictions are intentionally excluded.
    # -------------------------------------------------------------------------
    copy_eyeear_tables(
        run_dir,
        output_dir,
        manifest,
    )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------
    manifest_frame = pd.DataFrame(manifest)

    # Fixed journal/manuscript sequence, independent of table generation order.
    manifest_frame = sort_publication_manifest(
        manifest_frame
    )

    manifest_path = output_dir / "Publication_Table_Manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)

    # -------------------------------------------------------------------------
    # Word file containing all tables
    # -------------------------------------------------------------------------
    if not skip_tables_word:
        tables_word_path = (
            output_dir
            / "All_Tables_TemporalSafe_StrictOOF.docx"
        )
        try:
            build_tables_word(
                manifest_frame,
                tables_word_path,
                run_dir,
            )
            print("Tables Word file:", tables_word_path)
        except Exception as exc:
            print("WARNING: Could not create tables Word file:", exc)

    # -------------------------------------------------------------------------
    # Exact journal-submission figure package
    # -------------------------------------------------------------------------
    if not skip_figures_word:
        # Full archive remains useful internally.
        all_figure_paths = collect_current_run_figures(
            run_dir,
            output_dir,
        )

        # Submission set is deterministic and must contain exactly 12 figures.
        submission_figures, figure_manifest = resolve_submission_figure_set(
            run_dir,
            output_dir,
            figure1_path=figure1_path,
            figure2_path=figure2_path,
        )

        figure_manifest_path = (
            output_dir
            / "Figure_Submission_Manifest.csv"
        )
        figure_manifest.to_csv(
            figure_manifest_path,
            index=False,
        )

        submission_figures_word = (
            output_dir
            / "Submission_Figures_TemporalSafe_StrictOOF.docx"
        )

        archive_figures_word = (
            output_dir
            / "All_Figures_Archive_TemporalSafe_StrictOOF.docx"
        )

        build_figures_word(
            submission_figures,
            submission_figures_word,
            run_dir,
            document_title=(
                "Submission Figures — Temporal-Safe Strict-OOF Analysis"
            ),
        )

        print(
            "Submission figures Word file:",
            submission_figures_word,
        )
        print(
            "Submission figures selected:",
            len(submission_figures),
        )
        print(
            "Figure selection manifest:",
            figure_manifest_path,
        )

        # Full archive is internal only.
        build_figures_word(
            all_figure_paths,
            archive_figures_word,
            run_dir,
            document_title=(
                "All Figure Archive — NOT FOR JOURNAL SUBMISSION"
            ),
        )

        print(
            "Archive figures Word file:",
            archive_figures_word,
        )
        print(
            "All PNG figures archived:",
            len(all_figure_paths),
        )

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print("Audit model:", audit["model_name"])
    print("Audit probability column:", audit["probability_column"])
    print(f"Operating threshold: {threshold:.8f}")
    print("Table manifest:", manifest_path)
    print("\nTable status:")
    print(
        manifest_frame[
            ["table_id", "title", "status"]
        ].to_string(index=False)
    )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Build all publication tables from the final temporal-safe "
            "strict-OOF run and compile all current-run figures into Word."
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
        type=str,
        default="auto",
        help=(
            "Reference model for subgroup/error/calibration tables. "
            "Default 'auto' uses Hybrid DARN-XGB Blend."
        ),
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_JOBS,
    )
    parser.add_argument(
        "--n_calibration_bins",
        type=int,
        default=DEFAULT_CALIBRATION_BINS,
    )
    parser.add_argument(
        "--top_explainability_features",
        type=int,
        default=DEFAULT_TOP_EXPLAINABILITY_FEATURES,
    )
    parser.add_argument(
        "--figure1_path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to the final Figure 1 PNG. Required unless a canonical "
            "Figure1_Final.png has been copied into the final run directory."
        ),
    )
    parser.add_argument(
        "--figure2_path",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to the final Figure 2 PNG. Required unless a canonical "
            "Figure2_TemporalSafe_StrictOOF_Cohort_Characteristics.png has "
            "been copied into the final run directory."
        ),
    )
    parser.add_argument(
        "--skip_tables_word",
        action="store_true",
    )
    parser.add_argument(
        "--skip_figures_word",
        action="store_true",
    )

    raw = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    # Jupyter/IPython-safe parsing.
    filtered: List[str] = []
    skip_next = False

    for argument in raw:
        if skip_next:
            skip_next = False
            continue

        if argument in {"-f", "--f"}:
            skip_next = True
            continue

        if argument.startswith("-f=") or argument.startswith("--f="):
            continue

        filtered.append(argument)

    args, unknown = parser.parse_known_args(filtered)

    if unknown:
        print(
            "Ignoring unrecognized Jupyter/IPython arguments:",
            " ".join(map(str, unknown)),
        )

    return args


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.run_dir / DEFAULT_OUTPUT_SUBDIR
    )

    run_all(
        run_dir=args.run_dir,
        pipeline_module=args.pipeline_module,
        data_path=args.data_path,
        output_dir=output_dir,
        audit_model=args.audit_model,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        n_calibration_bins=args.n_calibration_bins,
        top_explainability_features=args.top_explainability_features,
        skip_tables_word=args.skip_tables_word,
        skip_figures_word=args.skip_figures_word,
        figure1_path=args.figure1_path,
        figure2_path=args.figure2_path,
    )


if __name__ == "__main__":
    main()
