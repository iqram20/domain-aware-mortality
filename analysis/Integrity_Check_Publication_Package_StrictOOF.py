#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrity check for the final temporal-safe + strict-OOF publication package.

The script does NOT compare against manually entered performance numbers.
Instead, it cross-checks the package against the completed run artifacts,
reconstructs the modeled cohort and saved test set, and verifies internal
consistency across tables, predictions, thresholds, Word files, and figures.

Default run
-----------
/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/
darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653

Outputs
-------
Integrity_Check_Report.csv
Integrity_Check_Summary.json
Figure_Integrity_Inventory.csv
Submission_Exclusion_List.csv

Exit code
---------
0 = no FAIL checks
1 = at least one FAIL check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import zipfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# =============================================================================
# DEFAULT FINAL-RUN PATHS
# =============================================================================

PROJECT_ROOT = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire"
)

DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "Results"
    / "darn_cv_baseline_runs"
    / "v6_temporal_safe_strict_oof_parallel"
    / "run_20260919_115653"
)

DEFAULT_PIPELINE_MODULE = (
    PROJECT_ROOT
    / "Code"
    / "New"
    / "Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py"
)

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "Results"
    / "Final_Datasets"
    / "df_final_mortality_2026-05-24_12-44_temporal_safe.csv"
)

DEFAULT_PACKAGE_SUBDIR = "publication_tables_temporal_safe_strict_oof"
DEFAULT_AUDIT_MODEL = "Hybrid DARN-XGB Blend"

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

OLD_RUN_MARKERS = [
    "v6_parallel/run_20260719_123200",
    "run_20260719_123200",
    "Mortality_DARN_full_pipe_v6_reviewer_complete.py",
    "df_final_mortality_2026-05-24_12-44.csv",
    "reviewer-v6",
    "current_v6",
    "locked holdout",
]

DIRECT_IDENTIFIER_PATTERNS = [
    r"^subject_?id$",
    r"^patient_?id$",
    r"^person_?id$",
    r"^op_?id$",
    r"^encounter_?id$",
    r"^visit_?id$",
    r"^mrn$",
    r"^name$",
    r"^dob$",
    r"^date_of_birth$",
]


# =============================================================================
# REPORTING HELPERS
# =============================================================================

class Reporter:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        check: str,
        status: str,
        detail: str,
        *,
        category: str = "General",
    ) -> None:
        status = status.upper()
        if status not in {"PASS", "WARN", "FAIL", "INFO"}:
            raise ValueError(f"Unsupported status: {status}")

        self.rows.append(
            {
                "category": category,
                "check": check,
                "status": status,
                "detail": str(detail),
            }
        )

        print(f"[{status:4s}] {category} | {check}: {detail}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def has_failures(self) -> bool:
        return any(row["status"] == "FAIL" for row in self.rows)

    def counts(self) -> Dict[str, int]:
        counter = defaultdict(int)
        for row in self.rows:
            counter[row["status"]] += 1
        return dict(counter)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


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


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with open(require_file(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pipeline_module(path: pathlib.Path):
    require_file(path)

    spec = importlib.util.spec_from_file_location(
        "strict_oof_integrity_pipeline",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hash_file(path: pathlib.Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def frames_equal_numeric_tolerant(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> Tuple[bool, str]:
    if list(left.columns) != list(right.columns):
        return False, (
            f"column mismatch: left={list(left.columns)}; "
            f"right={list(right.columns)}"
        )

    if len(left) != len(right):
        return False, f"row-count mismatch: {len(left)} vs {len(right)}"

    for column in left.columns:
        l = left[column]
        r = right[column]

        ln = pd.to_numeric(l, errors="coerce")
        rn = pd.to_numeric(r, errors="coerce")

        numeric_like = (
            ln.notna().sum() == l.notna().sum()
            and rn.notna().sum() == r.notna().sum()
        )

        if numeric_like:
            if not np.allclose(
                ln.to_numpy(dtype=float),
                rn.to_numpy(dtype=float),
                atol=atol,
                rtol=0.0,
                equal_nan=True,
            ):
                return False, f"numeric mismatch in column '{column}'"
        else:
            ls = l.fillna("").astype(str).to_numpy()
            rs = r.fillna("").astype(str).to_numpy()

            if not np.array_equal(ls, rs):
                return False, f"text mismatch in column '{column}'"

    return True, "exact/tolerance match"


def prediction_position_column(frame: pd.DataFrame) -> Optional[str]:
    return first_existing_column(
        frame,
        [
            "final_cohort_row_position",
            "original_row_position",
            "source_row_position",
        ],
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def almost_equal(
    observed: Any,
    expected: Any,
    *,
    atol: float = 1e-10,
) -> bool:
    try:
        a = float(observed)
        b = float(expected)
    except Exception:
        return str(observed) == str(expected)

    if np.isnan(a) and np.isnan(b):
        return True

    return bool(np.isclose(a, b, atol=atol, rtol=0.0))


# =============================================================================
# RUN / COHORT RECONSTRUCTION
# =============================================================================

def reconstruct_run(
    run_dir: pathlib.Path,
    pipeline_module_path: pathlib.Path,
    data_path: pathlib.Path,
    reporter: Reporter,
) -> Dict[str, Any]:
    pipeline = load_pipeline_module(pipeline_module_path)
    config = load_json(run_dir / "config.json")

    configured_data = pathlib.Path(
        str(config.get("data_path", data_path))
    )

    if configured_data.exists():
        same_data = configured_data.resolve() == data_path.resolve()
        reporter.add(
            "Config data path",
            "PASS" if same_data else "WARN",
            (
                f"config={configured_data}; requested={data_path}"
                if not same_data
                else str(configured_data)
            ),
            category="Provenance",
        )
    else:
        reporter.add(
            "Config data path exists",
            "WARN",
            f"Configured path does not exist: {configured_data}",
            category="Provenance",
        )

    source = pd.read_csv(require_file(data_path), low_memory=False)

    target_column = str(config.get("target_column", "mortality_30d"))

    cohort, cohort_flow = pipeline.apply_reviewer_cohort_exclusions(
        source,
        target_column=target_column,
        exclude_asa6=bool(config.get("exclude_asa6", True)),
    )
    cohort = cohort.reset_index(drop=True)

    target = pd.to_numeric(
        cohort[target_column],
        errors="raise",
    ).astype(int)

    test_path = run_dir / "test_predictions_all_models.csv"
    test_predictions = pd.read_csv(
        require_file(test_path),
        low_memory=False,
    )

    if "y_true" not in test_predictions.columns:
        raise KeyError("test_predictions_all_models.csv is missing y_true.")

    test_position_col = prediction_position_column(test_predictions)
    if test_position_col is None:
        raise KeyError("No row-position column in test predictions.")

    if normalize_name(test_position_col) == normalize_name("source_row_position"):
        source_positions = pd.to_numeric(
            test_predictions[test_position_col],
            errors="raise",
        ).astype(int).to_numpy()

        source_indexed = (
            source.reset_index(drop=False)
            .rename(columns={"index": "source_row_position"})
        )

        test_cohort = (
            source_indexed.set_index("source_row_position")
            .loc[source_positions]
            .reset_index(drop=True)
        )
        test_positions = None
    else:
        test_positions = pd.to_numeric(
            test_predictions[test_position_col],
            errors="raise",
        ).astype(int).to_numpy()

        test_cohort = (
            cohort.iloc[test_positions]
            .reset_index(drop=True)
        )

    y_test_saved = pd.to_numeric(
        test_predictions["y_true"],
        errors="raise",
    ).astype(int).to_numpy()

    y_test_reconstructed = pd.to_numeric(
        test_cohort[target_column],
        errors="raise",
    ).astype(int).to_numpy()

    reporter.add(
        "Saved test outcomes match reconstructed cohort",
        "PASS" if np.array_equal(y_test_saved, y_test_reconstructed) else "FAIL",
        (
            f"N={len(y_test_saved):,}; deaths={int(y_test_saved.sum()):,}"
        ),
        category="Cohort",
    )

    dev_path = None
    for candidate in [
        run_dir / "development_oof_predictions.csv",
        run_dir / "development_predictions.csv",
    ]:
        if candidate.exists():
            dev_path = candidate
            break

    development_predictions = None
    development_positions = None
    y_dev = None

    if dev_path is not None:
        development_predictions = pd.read_csv(dev_path, low_memory=False)
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

            y_dev = target.iloc[development_positions].to_numpy()

            if "y_true" in development_predictions.columns:
                y_dev_saved = pd.to_numeric(
                    development_predictions["y_true"],
                    errors="raise",
                ).astype(int).to_numpy()

                reporter.add(
                    "Saved development outcomes match cohort",
                    "PASS" if np.array_equal(y_dev, y_dev_saved) else "FAIL",
                    (
                        f"N={len(y_dev):,}; deaths={int(y_dev.sum()):,}"
                    ),
                    category="Cohort",
                )
    else:
        reporter.add(
            "Development prediction file",
            "WARN",
            "No development prediction file found.",
            category="Cohort",
        )

    if development_positions is not None and test_positions is not None:
        overlap = np.intersect1d(development_positions, test_positions)

        reporter.add(
            "Development/test overlap",
            "PASS" if len(overlap) == 0 else "FAIL",
            f"overlapping row positions={len(overlap):,}",
            category="Cohort",
        )

        represented = np.concatenate(
            [development_positions, test_positions]
        )
        unique_represented = np.unique(represented)

        exhaustive = (
            len(unique_represented) == len(cohort)
            and unique_represented.min(initial=0) == 0
            and unique_represented.max(initial=-1) == len(cohort) - 1
        )

        reporter.add(
            "Development + test exhaust modeled cohort",
            "PASS" if exhaustive else "FAIL",
            (
                f"represented={len(unique_represented):,}; "
                f"modeled cohort={len(cohort):,}"
            ),
            category="Cohort",
        )

        no_duplicates = len(represented) == len(unique_represented)
        reporter.add(
            "No duplicate split row positions",
            "PASS" if no_duplicates else "FAIL",
            (
                f"saved positions={len(represented):,}; "
                f"unique={len(unique_represented):,}"
            ),
            category="Cohort",
        )

    reporter.add(
        "Modeled cohort target validity",
        (
            "PASS"
            if set(target.unique()).issubset({0, 1}) and not target.isna().any()
            else "FAIL"
        ),
        (
            f"N={len(cohort):,}; deaths={int(target.sum()):,}; "
            f"unique target values={sorted(target.unique().tolist())}"
        ),
        category="Cohort",
    )

    return {
        "pipeline": pipeline,
        "config": config,
        "source": source,
        "cohort": cohort,
        "cohort_flow": cohort_flow,
        "target_column": target_column,
        "target": target,
        "test_predictions": test_predictions,
        "test_cohort": test_cohort,
        "test_positions": test_positions,
        "y_test": y_test_saved,
        "development_predictions": development_predictions,
        "development_positions": development_positions,
        "y_dev": y_dev,
    }


# =============================================================================
# MODEL / THRESHOLD RESOLUTION
# =============================================================================

def match_model_row(
    metrics: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    model_col = first_existing_column(metrics, ["model"])
    if model_col is None:
        raise KeyError("Metrics table lacks model column.")

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

    raise KeyError(f"Model not found: {model_name}")


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
        raise KeyError("No probability/risk column found.")

    aliases = MODEL_ALIASES.get(
        model_name,
        [normalize_name(model_name)],
    )

    scored: List[Tuple[float, str]] = []

    for column in candidates:
        normalized = normalize_name(column)
        score = 0.0

        for alias in aliases:
            if alias in normalized:
                score += 20.0 + len(alias) / 100.0

        if "calibrated" in normalized:
            score += 5.0

        if "probability" in normalized:
            score += 2.0

        if (
            model_name == "Hybrid DARN-XGB Blend"
            and "hybrid" not in normalized
            and "blend" not in normalized
        ):
            score -= 10.0

        scored.append((score, column))

    scored.sort(reverse=True)
    best_score, best_column = scored[0]

    if best_score <= 0:
        raise KeyError(
            f"Could not resolve probability column for {model_name}."
        )

    return best_column


def resolve_audit_model(
    run_dir: pathlib.Path,
    predictions: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    metrics = pd.read_csv(
        require_file(run_dir / "test_model_metrics.csv")
    )

    requested = (
        DEFAULT_AUDIT_MODEL
        if str(model_name).strip().lower() in {"", "auto"}
        else str(model_name)
    )

    row = match_model_row(metrics, requested)

    model_col = first_existing_column(metrics, ["model"])
    threshold_col = first_existing_column(
        metrics,
        ["Threshold", "threshold"],
    )

    if model_col is None or threshold_col is None:
        raise KeyError("Model or threshold column missing from test metrics.")

    actual_model = str(row[model_col])
    probability_col = resolve_probability_column(
        predictions,
        actual_model,
    )

    probability = pd.to_numeric(
        predictions[probability_col],
        errors="raise",
    ).to_numpy(dtype=float)

    return {
        "metrics": metrics,
        "row": row,
        "model_name": actual_model,
        "threshold": float(row[threshold_col]),
        "probability_column": probability_col,
        "probability": probability,
    }


def point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predicted = (p >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))

    result = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "PPV": safe_divide(tp, tp + fp),
        "NPV": safe_divide(tn, tn + fn),
        "FPR": safe_divide(fp, fp + tn),
        "Brier": float(brier_score_loss(y, p)),
        "AUROC": (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "AUPRC": (
            float(average_precision_score(y, p))
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "observed_expected_ratio": safe_divide(
            float(y.sum()),
            float(p.sum()),
        ),
    }

    return result


# =============================================================================
# MANIFEST / TABLE CHECKS
# =============================================================================

def check_manifest(
    package_dir: pathlib.Path,
    reporter: Reporter,
) -> pd.DataFrame:
    manifest_path = package_dir / "Publication_Table_Manifest.csv"

    if not manifest_path.exists():
        reporter.add(
            "Publication table manifest",
            "FAIL",
            f"Missing: {manifest_path}",
            category="Tables",
        )
        return pd.DataFrame()

    manifest = pd.read_csv(manifest_path)

    reporter.add(
        "Publication table manifest",
        "PASS",
        f"rows={len(manifest):,}",
        category="Tables",
    )

    required_cols = {"table_id", "title", "file", "source", "status"}

    reporter.add(
        "Manifest schema",
        "PASS" if required_cols.issubset(manifest.columns) else "FAIL",
        f"columns={list(manifest.columns)}",
        category="Tables",
    )

    if not required_cols.issubset(manifest.columns):
        return manifest

    created = manifest.loc[
        manifest["status"].astype(str).str.lower().eq("created")
    ]

    missing_files = []

    for row in created.itertuples(index=False):
        path_text = str(getattr(row, "file"))
        if not path_text or path_text.lower() == "nan":
            missing_files.append(
                f"{getattr(row, 'table_id')}: blank path"
            )
            continue

        if not pathlib.Path(path_text).exists():
            missing_files.append(
                f"{getattr(row, 'table_id')}: {path_text}"
            )

    reporter.add(
        "All manifest-created files exist",
        "PASS" if not missing_files else "FAIL",
        (
            "all files present"
            if not missing_files
            else " | ".join(missing_files)
        ),
        category="Tables",
    )

    duplicate_ids = manifest["table_id"].astype(str).duplicated().sum()

    reporter.add(
        "Manifest table IDs unique",
        "PASS" if duplicate_ids == 0 else "WARN",
        f"duplicate IDs={int(duplicate_ids)}",
        category="Tables",
    )

    return manifest


def check_s3_exact(
    package_dir: pathlib.Path,
    run_dir: pathlib.Path,
    reporter: Reporter,
) -> None:
    table_path = package_dir / "TableS3_Full_HeldOut_Test_Model_Metrics.csv"
    source_path = run_dir / "test_model_metrics.csv"

    if not table_path.exists() or not source_path.exists():
        reporter.add(
            "Table S3 exact source match",
            "FAIL",
            f"Missing table or source: {table_path} | {source_path}",
            category="Tables",
        )
        return

    table = pd.read_csv(table_path)
    source = pd.read_csv(source_path)

    equal, detail = frames_equal_numeric_tolerant(table, source)

    reporter.add(
        "Table S3 exact source match",
        "PASS" if equal else "FAIL",
        detail,
        category="Tables",
    )


def check_table2_models(
    package_dir: pathlib.Path,
    run_dir: pathlib.Path,
    reporter: Reporter,
) -> None:
    table_path = package_dir / "Table2_HeldOut_Test_Model_Performance.csv"
    metrics_path = run_dir / "test_model_metrics.csv"

    if not table_path.exists() or not metrics_path.exists():
        reporter.add(
            "Table 2 model coverage",
            "FAIL",
            "Table 2 or test_model_metrics.csv missing.",
            category="Tables",
        )
        return

    table = pd.read_csv(table_path)
    metrics = pd.read_csv(metrics_path)

    table_model_col = first_existing_column(table, ["Model"])
    metrics_model_col = first_existing_column(metrics, ["model"])

    if table_model_col is None or metrics_model_col is None:
        reporter.add(
            "Table 2 model coverage",
            "FAIL",
            "Model column missing.",
            category="Tables",
        )
        return

    table_models = set(
        table[table_model_col].astype(str).map(normalize_name)
    )
    metric_models = set(
        metrics[metrics_model_col].astype(str).map(normalize_name)
    )

    reporter.add(
        "Table 2 model coverage",
        "PASS" if table_models == metric_models else "FAIL",
        (
            f"Table2 models={len(table_models)}; "
            f"metrics models={len(metric_models)}; "
            f"missing={sorted(metric_models - table_models)}; "
            f"extra={sorted(table_models - metric_models)}"
        ),
        category="Tables",
    )


def check_s1_consistency(
    package_dir: pathlib.Path,
    reconstructed: Mapping[str, Any],
    reporter: Reporter,
) -> None:
    split_path = package_dir / "TableS1b_Development_Test_Split_Audit.csv"

    if not split_path.exists():
        reporter.add(
            "Table S1b split consistency",
            "FAIL",
            "Table S1b missing.",
            category="Tables",
        )
        return

    split = pd.read_csv(split_path)

    cohort_col = first_existing_column(split, ["Cohort"])
    n_col = first_existing_column(split, ["Encounters"])
    deaths_col = first_existing_column(split, ["Deaths"])

    if cohort_col is None or n_col is None or deaths_col is None:
        reporter.add(
            "Table S1b split consistency",
            "FAIL",
            "Required columns absent.",
            category="Tables",
        )
        return

    mapping = {
        normalize_name(row[cohort_col]): row
        for _, row in split.iterrows()
    }

    expected: Dict[str, Tuple[int, int]] = {
        "heldouttest": (
            len(reconstructed["y_test"]),
            int(reconstructed["y_test"].sum()),
        ),
        "overallmodeledcohort": (
            len(reconstructed["cohort"]),
            int(reconstructed["target"].sum()),
        ),
    }

    if reconstructed["y_dev"] is not None:
        expected["development"] = (
            len(reconstructed["y_dev"]),
            int(reconstructed["y_dev"].sum()),
        )

    mismatches = []

    for key, (n_expected, d_expected) in expected.items():
        row = mapping.get(key)

        if row is None:
            mismatches.append(f"missing row: {key}")
            continue

        n_observed = int(row[n_col])
        d_observed = int(row[deaths_col])

        if n_observed != n_expected or d_observed != d_expected:
            mismatches.append(
                f"{key}: observed {n_observed}/{d_observed}; "
                f"expected {n_expected}/{d_expected}"
            )

    reporter.add(
        "Table S1b split consistency",
        "PASS" if not mismatches else "FAIL",
        (
            "matches reconstructed split"
            if not mismatches
            else " | ".join(mismatches)
        ),
        category="Tables",
    )


def check_s7_overall(
    package_dir: pathlib.Path,
    audit: Mapping[str, Any],
    y_test: np.ndarray,
    reporter: Reporter,
) -> None:
    path = package_dir / "TableS7_Subgroup_Performance_and_Operating_Point_Audit.csv"

    if not path.exists():
        reporter.add(
            "Table S7 overall row",
            "FAIL",
            "Table S7 missing.",
            category="Tables",
        )
        return

    table = pd.read_csv(path)

    variable_col = first_existing_column(table, ["subgroup_variable"])
    level_col = first_existing_column(table, ["subgroup_level"])

    if variable_col is None or level_col is None:
        reporter.add(
            "Table S7 overall row",
            "FAIL",
            "Subgroup columns missing.",
            category="Tables",
        )
        return

    overall = table.loc[
        table[variable_col].astype(str).eq("Overall")
        & table[level_col].astype(str).eq("Overall")
    ]

    if len(overall) != 1:
        reporter.add(
            "Table S7 overall row",
            "FAIL",
            f"Expected exactly one Overall row; found {len(overall)}",
            category="Tables",
        )
        return

    row = overall.iloc[0]
    expected = point_metrics(
        y_test,
        audit["probability"],
        audit["threshold"],
    )

    fields = [
        "n",
        "events",
        "prevalence",
        "AUROC",
        "AUPRC",
        "Brier",
        "sensitivity",
        "specificity",
        "PPV",
        "FPR",
        "observed_expected_ratio",
    ]

    mismatches = []

    for field in fields:
        column = first_existing_column(table, [field])

        if column is None:
            continue

        if not almost_equal(row[column], expected[field], atol=1e-10):
            mismatches.append(
                f"{field}: table={row[column]}; recomputed={expected[field]}"
            )

    threshold_col = first_existing_column(
        table,
        ["operating_threshold"],
    )
    if threshold_col is not None and not almost_equal(
        row[threshold_col],
        audit["threshold"],
        atol=1e-12,
    ):
        mismatches.append(
            f"threshold: table={row[threshold_col]}; "
            f"saved={audit['threshold']}"
        )

    reporter.add(
        "Table S7 overall row",
        "PASS" if not mismatches else "FAIL",
        (
            "point estimates exactly reproduce saved test predictions"
            if not mismatches
            else " | ".join(mismatches)
        ),
        category="Tables",
    )


def check_s8_calibration(
    package_dir: pathlib.Path,
    y_test: np.ndarray,
    reporter: Reporter,
) -> None:
    path = package_dir / "TableS8_Tail_Calibration_Quantile_Bins.csv"

    if not path.exists():
        reporter.add(
            "Table S8 calibration totals",
            "FAIL",
            "Table S8 missing.",
            category="Tables",
        )
        return

    table = pd.read_csv(path)

    n_col = first_existing_column(table, ["n"])
    events_col = first_existing_column(table, ["events"])

    if n_col is None or events_col is None:
        reporter.add(
            "Table S8 calibration totals",
            "FAIL",
            "n/events columns missing.",
            category="Tables",
        )
        return

    n_total = int(pd.to_numeric(table[n_col], errors="raise").sum())
    events_total = int(
        pd.to_numeric(table[events_col], errors="raise").sum()
    )

    passed = (
        n_total == len(y_test)
        and events_total == int(y_test.sum())
    )

    reporter.add(
        "Table S8 calibration totals",
        "PASS" if passed else "FAIL",
        (
            f"bins N/deaths={n_total:,}/{events_total:,}; "
            f"test N/deaths={len(y_test):,}/{int(y_test.sum()):,}"
        ),
        category="Tables",
    )


def check_s9_s10_errors(
    package_dir: pathlib.Path,
    audit: Mapping[str, Any],
    y_test: np.ndarray,
    reporter: Reporter,
) -> None:
    s9_path = package_dir / "TableS9_Error_Group_Clinical_Profiles.csv"
    s10_path = package_dir / "TableS10_Deidentified_False_Negative_Review.csv"

    expected = point_metrics(
        y_test,
        audit["probability"],
        audit["threshold"],
    )

    expected_counts = {
        "True positive": expected["TP"],
        "False positive": expected["FP"],
        "False negative": expected["FN"],
        "True negative": expected["TN"],
    }

    if s9_path.exists():
        s9 = pd.read_csv(s9_path)
        group_col = first_existing_column(s9, ["error_group"])
        n_col = first_existing_column(s9, ["n"])

        mismatches = []

        if group_col is None or n_col is None:
            mismatches.append("error_group/n columns missing")
        else:
            mapping = {
                str(row[group_col]): int(row[n_col])
                for _, row in s9.iterrows()
            }

            for group, count in expected_counts.items():
                if mapping.get(group) != count:
                    mismatches.append(
                        f"{group}: table={mapping.get(group)}; expected={count}"
                    )

        reporter.add(
            "Table S9 confusion counts",
            "PASS" if not mismatches else "FAIL",
            (
                "TP/FP/FN/TN counts reproduce thresholded test predictions"
                if not mismatches
                else " | ".join(mismatches)
            ),
            category="Tables",
        )
    else:
        reporter.add(
            "Table S9 confusion counts",
            "FAIL",
            "Table S9 missing.",
            category="Tables",
        )

    if s10_path.exists():
        s10 = pd.read_csv(s10_path)

        reporter.add(
            "Table S10 false-negative row count",
            "PASS" if len(s10) == expected["FN"] else "FAIL",
            f"rows={len(s10)}; expected FN={expected['FN']}",
            category="Tables",
        )

        bad_columns = []

        for column in s10.columns:
            norm = normalize_name(column)
            for pattern in DIRECT_IDENTIFIER_PATTERNS:
                if re.fullmatch(pattern, norm, flags=re.IGNORECASE):
                    bad_columns.append(column)
                    break

        reporter.add(
            "Table S10 direct-identifier columns",
            "PASS" if not bad_columns else "FAIL",
            (
                "none detected"
                if not bad_columns
                else f"detected={bad_columns}"
            ),
            category="Privacy",
        )
    else:
        reporter.add(
            "Table S10 false-negative review",
            "FAIL",
            "Table S10 missing.",
            category="Tables",
        )


def check_s11(
    package_dir: pathlib.Path,
    run_dir: pathlib.Path,
    reporter: Reporter,
) -> None:
    path = package_dir / "TableS11_Component_Explainability_Features.csv"

    if not path.exists():
        reporter.add(
            "Table S11 explainability provenance",
            "FAIL",
            "Table S11 missing.",
            category="Tables",
        )
        return

    table = pd.read_csv(path)

    component_col = first_existing_column(table, ["component"])

    if component_col is None:
        reporter.add(
            "Table S11 explainability provenance",
            "FAIL",
            "component column missing.",
            category="Tables",
        )
        return

    components = set(table[component_col].astype(str))

    fig4_dir = (
        run_dir
        / "figures_revised_manuscript"
        / "figure4_darn_xgboost_AB_strict_oof"
    )

    expected_components = set()

    if (fig4_dir / "uniform_darn_gradientshap_feature_importance.csv").exists():
        expected_components.add("Uniform DARN")

    if (fig4_dir / "xgboost_treeshap_feature_importance.csv").exists():
        expected_components.add("XGBoost")

    reporter.add(
        "Table S11 explainability provenance",
        "PASS" if components == expected_components else "FAIL",
        (
            f"table components={sorted(components)}; "
            f"available Figure4 components={sorted(expected_components)}"
        ),
        category="Tables",
    )


def check_s12_submission_exclusions(
    package_dir: pathlib.Path,
    manifest: pd.DataFrame,
    reporter: Reporter,
) -> pd.DataFrame:
    exclusions: List[Dict[str, Any]] = []

    if manifest.empty:
        return pd.DataFrame(exclusions)

    for row in manifest.itertuples(index=False):
        table_id = str(getattr(row, "table_id", ""))
        title = str(getattr(row, "title", ""))
        path_text = str(getattr(row, "file", ""))

        normalized_title = normalize_name(title)
        normalized_path = normalize_name(path_text)

        if (
            "testsprobabilities" in normalized_title
            or "testprobabilities" in normalized_title
            or "testprobabilities" in normalized_path
        ):
            exclusions.append(
                {
                    "artifact": path_text,
                    "reason": (
                        "Row-level prediction artifact; retain for analysis/audit "
                        "but do not include as a manuscript supplementary table."
                    ),
                    "table_id": table_id,
                }
            )

    reporter.add(
        "Row-level prediction artifacts excluded from submission",
        "WARN" if exclusions else "PASS",
        (
            f"{len(exclusions)} artifact(s) should be excluded"
            if exclusions
            else "none identified"
        ),
        category="Submission",
    )

    return pd.DataFrame(exclusions)


# =============================================================================
# OLD-RUN CONTAMINATION SCAN
# =============================================================================

def text_from_docx(path: pathlib.Path) -> str:
    chunks: List[str] = []

    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                try:
                    chunks.append(
                        archive.read(name).decode("utf-8", errors="ignore")
                    )
                except Exception:
                    pass

    return "\n".join(chunks)


def contamination_scan(
    package_dir: pathlib.Path,
    reporter: Reporter,
) -> None:
    findings: List[str] = []

    scan_paths: List[pathlib.Path] = []

    for extension in ["*.csv", "*.json", "*.txt", "*.docx"]:
        scan_paths.extend(package_dir.rglob(extension))

    for path in sorted(set(scan_paths)):
        try:
            if path.suffix.lower() == ".docx":
                text = text_from_docx(path)
            else:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
        except Exception:
            continue

        lower = text.lower()

        for marker in OLD_RUN_MARKERS:
            if marker.lower() in lower:
                findings.append(
                    f"{path.name}: '{marker}'"
                )

    reporter.add(
        "Old-run terminology/path contamination",
        "PASS" if not findings else "FAIL",
        (
            "none detected"
            if not findings
            else " | ".join(findings[:20])
        ),
        category="Provenance",
    )


# =============================================================================
# WORD + FIGURE CHECKS
# =============================================================================

def check_docx(
    path: pathlib.Path,
    reporter: Reporter,
    *,
    label: str,
    expect_images: bool,
) -> Optional[int]:
    if not path.exists():
        reporter.add(
            label,
            "FAIL",
            f"Missing: {path}",
            category="Word",
        )
        return None

    try:
        with zipfile.ZipFile(path, "r") as archive:
            corrupt_member = archive.testzip()

            if corrupt_member is not None:
                reporter.add(
                    f"{label} ZIP integrity",
                    "FAIL",
                    f"Corrupt member: {corrupt_member}",
                    category="Word",
                )
                return None

            names = archive.namelist()
            media = [
                name
                for name in names
                if name.startswith("word/media/")
            ]

            document_xml = (
                archive.read("word/document.xml")
                .decode("utf-8", errors="ignore")
            )

            table_count = document_xml.count("<w:tbl")

            reporter.add(
                f"{label} ZIP integrity",
                "PASS",
                (
                    f"valid DOCX; size={path.stat().st_size:,} bytes; "
                    f"tables={table_count}; embedded media={len(media)}"
                ),
                category="Word",
            )

            if expect_images:
                reporter.add(
                    f"{label} contains images",
                    "PASS" if len(media) > 0 else "FAIL",
                    f"embedded media={len(media)}",
                    category="Word",
                )

            return len(media)
    except Exception as exc:
        reporter.add(
            label,
            "FAIL",
            f"DOCX validation error: {exc}",
            category="Word",
        )
        return None


def collect_source_pngs(
    run_dir: pathlib.Path,
    package_dir: pathlib.Path,
) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []

    for path in run_dir.rglob("*.png"):
        try:
            path.relative_to(package_dir)
            continue
        except ValueError:
            pass

        paths.append(path)

    return sorted(set(paths))


def figure_inventory(
    run_dir: pathlib.Path,
    package_dir: pathlib.Path,
    reporter: Reporter,
) -> pd.DataFrame:
    paths = collect_source_pngs(run_dir, package_dir)

    rows: List[Dict[str, Any]] = []

    for path in paths:
        digest = hash_file(path)

        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": digest,
            }
        )

    inventory = pd.DataFrame(rows)

    if inventory.empty:
        reporter.add(
            "Current-run PNG inventory",
            "FAIL",
            "No PNG figures found.",
            category="Figures",
        )
        return inventory

    duplicate_hashes = (
        inventory["sha256"]
        .value_counts()
        .loc[lambda series: series > 1]
    )

    duplicate_files = int(
        inventory["sha256"]
        .isin(set(duplicate_hashes.index))
        .sum()
    )

    reporter.add(
        "Current-run PNG inventory",
        "PASS",
        f"PNG files discovered={len(inventory):,}",
        category="Figures",
    )

    reporter.add(
        "Exact duplicate PNG files",
        "WARN" if duplicate_files else "PASS",
        (
            f"{duplicate_files} files fall into "
            f"{len(duplicate_hashes)} duplicate hash group(s)"
            if duplicate_files
            else "none"
        ),
        category="Figures",
    )

    if len(inventory) > 15:
        reporter.add(
            "Figure Word submission scope",
            "WARN",
            (
                f"{len(inventory)} PNGs exist in the run. "
                "This likely includes intermediate/audit plots; curate the "
                "submission figure document rather than submitting all."
            ),
            category="Submission",
        )
    else:
        reporter.add(
            "Figure Word submission scope",
            "PASS",
            f"{len(inventory)} PNGs found.",
            category="Submission",
        )

    return inventory


# =============================================================================
# MAIN
# =============================================================================

def run_integrity_check(
    run_dir: pathlib.Path,
    pipeline_module: pathlib.Path,
    data_path: pathlib.Path,
    package_dir: pathlib.Path,
    *,
    audit_model: str,
) -> int:
    reporter = Reporter()

    print("=" * 110)
    print("FINAL TEMPORAL-SAFE STRICT-OOF PUBLICATION PACKAGE — INTEGRITY CHECK")
    print("=" * 110)
    print("Run directory :", run_dir)
    print("Pipeline      :", pipeline_module)
    print("Data          :", data_path)
    print("Package       :", package_dir)
    print()

    # -------------------------------------------------------------------------
    # Provenance existence / naming
    # -------------------------------------------------------------------------
    for label, path in [
        ("Run directory", run_dir),
        ("Pipeline module", pipeline_module),
        ("Temporal-safe dataset", data_path),
        ("Publication package directory", package_dir),
    ]:
        reporter.add(
            label,
            "PASS" if path.exists() else "FAIL",
            str(path),
            category="Provenance",
        )

    reporter.add(
        "Strict-OOF run naming",
        "PASS" if "strict_oof" in str(run_dir).lower() else "WARN",
        run_dir.name,
        category="Provenance",
    )

    reporter.add(
        "Temporal-safe dataset naming",
        "PASS" if "temporal_safe" in data_path.name.lower() else "FAIL",
        data_path.name,
        category="Provenance",
    )

    reporter.add(
        "Strict-OOF pipeline naming",
        "PASS" if "strict_oof" in pipeline_module.name.lower() else "FAIL",
        pipeline_module.name,
        category="Provenance",
    )

    # Stop only if core inputs are absent.
    if not (
        run_dir.exists()
        and pipeline_module.exists()
        and data_path.exists()
        and package_dir.exists()
    ):
        report = reporter.frame()
        package_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(
            package_dir / "Integrity_Check_Report.csv",
            index=False,
        )
        return 1

    # -------------------------------------------------------------------------
    # Reconstruct cohort / splits
    # -------------------------------------------------------------------------
    reconstructed = reconstruct_run(
        run_dir,
        pipeline_module,
        data_path,
        reporter,
    )

    # -------------------------------------------------------------------------
    # Threshold + probability consistency
    # -------------------------------------------------------------------------
    audit = resolve_audit_model(
        run_dir,
        reconstructed["test_predictions"],
        audit_model,
    )

    reporter.add(
        "Audit model resolved",
        "PASS",
        (
            f"{audit['model_name']} | "
            f"probability={audit['probability_column']} | "
            f"threshold={audit['threshold']:.10f}"
        ),
        category="Model",
    )

    reporter.add(
        "Audit probability length",
        (
            "PASS"
            if len(audit["probability"]) == len(reconstructed["y_test"])
            else "FAIL"
        ),
        (
            f"probabilities={len(audit['probability']):,}; "
            f"test outcomes={len(reconstructed['y_test']):,}"
        ),
        category="Model",
    )

    reporter.add(
        "Audit probabilities finite and bounded",
        (
            "PASS"
            if (
                np.isfinite(audit["probability"]).all()
                and np.all(audit["probability"] >= 0)
                and np.all(audit["probability"] <= 1)
            )
            else "FAIL"
        ),
        (
            f"min={float(np.nanmin(audit['probability'])):.8g}; "
            f"max={float(np.nanmax(audit['probability'])):.8g}"
        ),
        category="Model",
    )

    recomputed = point_metrics(
        reconstructed["y_test"],
        audit["probability"],
        audit["threshold"],
    )

    metric_row = audit["row"]
    metrics_table = audit["metrics"]

    metric_checks = []

    for logical_name, candidates in [
        ("AUROC", ["AUROC"]),
        ("AUPRC", ["AUPRC"]),
        ("Brier", ["Brier", "brier"]),
        ("Sensitivity", ["Sensitivity", "sensitivity"]),
        ("Specificity", ["Specificity", "specificity"]),
        ("PPV", ["PPV", "precision_PPV", "positive_predictive_value"]),
    ]:
        column = first_existing_column(metrics_table, candidates)

        if column is None:
            continue

        expected_key = {
            "Sensitivity": "sensitivity",
            "Specificity": "specificity",
            "PPV": "PPV",
        }.get(logical_name, logical_name)

        if not almost_equal(
            metric_row[column],
            recomputed[expected_key],
            atol=1e-10,
        ):
            metric_checks.append(
                f"{logical_name}: saved={metric_row[column]}; "
                f"recomputed={recomputed[expected_key]}"
            )

    reporter.add(
        "Saved audit-model test metrics reproduce predictions",
        "PASS" if not metric_checks else "FAIL",
        (
            "AUROC/AUPRC/Brier/operating-point metrics consistent"
            if not metric_checks
            else " | ".join(metric_checks)
        ),
        category="Model",
    )

    # -------------------------------------------------------------------------
    # Table package
    # -------------------------------------------------------------------------
    manifest = check_manifest(package_dir, reporter)
    check_s3_exact(package_dir, run_dir, reporter)
    check_table2_models(package_dir, run_dir, reporter)
    check_s1_consistency(package_dir, reconstructed, reporter)
    check_s7_overall(
        package_dir,
        audit,
        reconstructed["y_test"],
        reporter,
    )
    check_s8_calibration(
        package_dir,
        reconstructed["y_test"],
        reporter,
    )
    check_s9_s10_errors(
        package_dir,
        audit,
        reconstructed["y_test"],
        reporter,
    )
    check_s11(package_dir, run_dir, reporter)

    exclusions = check_s12_submission_exclusions(
        package_dir,
        manifest,
        reporter,
    )

    contamination_scan(package_dir, reporter)

    # -------------------------------------------------------------------------
    # Word files
    # -------------------------------------------------------------------------
    tables_word = (
        package_dir
        / "All_Tables_TemporalSafe_StrictOOF.docx"
    )
    figures_word = (
        package_dir
        / "All_Figures_TemporalSafe_StrictOOF.docx"
    )

    check_docx(
        tables_word,
        reporter,
        label="All-tables Word file",
        expect_images=False,
    )

    embedded_figure_count = check_docx(
        figures_word,
        reporter,
        label="All-figures Word file",
        expect_images=True,
    )

    # -------------------------------------------------------------------------
    # Figure inventory / duplicates
    # -------------------------------------------------------------------------
    inventory = figure_inventory(
        run_dir,
        package_dir,
        reporter,
    )

    if embedded_figure_count is not None and not inventory.empty:
        reporter.add(
            "Word embedded-image count vs source PNG count",
            (
                "PASS"
                if embedded_figure_count == len(inventory)
                else "WARN"
            ),
            (
                f"Word media={embedded_figure_count}; "
                f"source PNG inventory={len(inventory)}"
            ),
            category="Figures",
        )

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------
    report_frame = reporter.frame()

    report_path = package_dir / "Integrity_Check_Report.csv"
    report_frame.to_csv(report_path, index=False)

    inventory_path = package_dir / "Figure_Integrity_Inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    exclusion_path = package_dir / "Submission_Exclusion_List.csv"

    if exclusions.empty:
        exclusions = pd.DataFrame(
            columns=["artifact", "reason", "table_id"]
        )

    exclusions.to_csv(exclusion_path, index=False)

    summary = {
        "run_directory": str(run_dir),
        "pipeline_module": str(pipeline_module),
        "data_path": str(data_path),
        "package_directory": str(package_dir),
        "audit_model": audit["model_name"],
        "audit_probability_column": audit["probability_column"],
        "audit_threshold": float(audit["threshold"]),
        "modeled_cohort_n": int(len(reconstructed["cohort"])),
        "modeled_cohort_events": int(reconstructed["target"].sum()),
        "test_n": int(len(reconstructed["y_test"])),
        "test_events": int(reconstructed["y_test"].sum()),
        "development_n": (
            None
            if reconstructed["y_dev"] is None
            else int(len(reconstructed["y_dev"]))
        ),
        "development_events": (
            None
            if reconstructed["y_dev"] is None
            else int(reconstructed["y_dev"].sum())
        ),
        "check_counts": reporter.counts(),
        "has_failures": reporter.has_failures(),
        "source_png_count": int(len(inventory)),
        "submission_exclusion_count": int(len(exclusions)),
        "submission_ready_from_integrity_checks": (
            not reporter.has_failures()
        ),
        "notes": [
            (
                "WARN does not necessarily indicate numerical error. "
                "Submission-scope warnings identify audit/intermediate artifacts "
                "that should be curated before journal upload."
            ),
            (
                "Row-level test-probability files should remain analysis artifacts "
                "rather than manuscript supplementary tables."
            ),
        ],
    }

    summary_path = package_dir / "Integrity_Check_Summary.json"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 110)
    print("INTEGRITY CHECK COMPLETE")
    print("=" * 110)
    print("Report       :", report_path)
    print("Summary      :", summary_path)
    print("Figure audit :", inventory_path)
    print("Exclusions   :", exclusion_path)
    print("Check counts :", reporter.counts())
    print(
        "FINAL STATUS :",
        "FAIL — review failed checks"
        if reporter.has_failures()
        else "PASS — no integrity failures detected",
    )

    return 1 if reporter.has_failures() else 0


# =============================================================================
# CLI
# =============================================================================

def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Cross-check the temporal-safe strict-OOF publication tables, "
            "Word files, figures, predictions, and run provenance."
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
        "--package_dir",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--audit_model",
        type=str,
        default="auto",
    )

    raw = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

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

    package_dir = (
        args.package_dir
        if args.package_dir is not None
        else args.run_dir / DEFAULT_PACKAGE_SUBDIR
    )

    exit_code = run_integrity_check(
        run_dir=args.run_dir,
        pipeline_module=args.pipeline_module,
        data_path=args.data_path,
        package_dir=package_dir,
        audit_model=args.audit_model,
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
