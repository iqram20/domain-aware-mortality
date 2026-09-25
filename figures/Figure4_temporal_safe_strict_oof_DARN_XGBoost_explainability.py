#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIGURE 4 — CLINICALLY LABELED DARN AND XGBOOST EXPLAINABILITY
=============================================

This standalone post-run script explains the Uniform DARN component from the
final temporal-safe, strict-OOF run:

    /athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/
    darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653

It does not retrain any model and does not use an XGBoost surrogate.

Why this script differs from the earlier explainability script
---------------------------------------------------------------
The final strict-OOF analysis uses five fold-specific Uniform DARN ensembles, with fold-specific
preprocessing and feature selection. Therefore, this script:

1. Reconstructs the exact ASA-6-excluded cohort and saved test ordering.
2. Loads every fold-specific preprocessor, variance selector, feature set,
   domain map and saved Uniform DARN state.
3. Computes direct GradientSHAP values for the same test sample in every fold.
4. Aligns transformed features by name across folds.
5. Averages signed attributions across the five-model DARN ensemble.
6. Produces global, patient-level and domain-level explanations.

The resulting attributions explain the Uniform DARN neural component. They do
not, by themselves, explain the complete DARN-XGBoost hybrid.

Outputs
-------
- Fig4A_Uniform_DARN_Direct_GradientSHAP_Beeswarm.*
- Fig4B_Uniform_DARN_Global_Feature_Importance.*
- Fig4C_Uniform_DARN_Patient_Level_Attribution.*
- Fig4D_Uniform_DARN_Domain_Attribution.*
- direct_uniform_darn_feature_importance.csv
- direct_uniform_darn_domain_importance.csv
- direct_uniform_darn_selected_patient.csv
- direct_uniform_darn_attribution_sample.csv
- direct_uniform_darn_ensemble_attributions.npz
- direct_uniform_darn_explainability_summary.json

The script is compatible with command-line execution and Jupyter `%run`. Missingness-indicator features are retained in complete CSV outputs but excluded from manuscript-facing feature rankings and beeswarm panels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# =============================================================================
# DEFAULT CURRENT-RUN PATHS
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

CMAP_NAME = "plasma"
RANDOM_STATE = 42
TOP_K = 25
MAIN_TOP_K = 15
PATIENT_TOP_K = 12
DISPLAY_CLIP_PERCENTILE = 99.5
MAX_EXPLAIN_N = 1000
BACKGROUND_N = 128
ATTRIBUTION_BATCH_SIZE = 64
GRADIENT_SAMPLES = 12

# Missingness indicators remain in the complete attribution CSVs for
# transparency, but they are excluded from the manuscript-facing beeswarm
# panels and console-ranked feature lists.
EXCLUDE_MISSING_INDICATORS_FROM_DISPLAY = True

BEESWARM_JITTER = 0.50
BEESWARM_DOT_SIZE = 10
BEESWARM_ALPHA = 0.72


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired current-run artifact was not found:\n{path}\n"
            "Confirm RUN_DIR, PIPELINE_MODULE and DATA_PATH."
        )
    return path


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with open(require_file(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Mapping[str, Any], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def load_pipeline_module(module_path: pathlib.Path):
    require_file(module_path)
    spec = importlib.util.spec_from_file_location(
        "mortality_darn_v6_explainability_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: pathlib.Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_object_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=object).astype(str)


def compress_words(text: str, front: int = 4, back: int = 2) -> str:
    words = str(text).split()
    if len(words) <= front + back + 1:
        return str(text)
    return " ".join(words[:front]) + " … " + " ".join(words[-back:])


# Short, publication-ready labels. Keys are lowercase so the lookup remains
# robust to capitalization differences in transformed feature names.
SHORT_LABEL_MAP = {
    # Demographics and case context
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "bmi": "BMI",
    "asa": "ASA class",
    "emop": "Emergency surgery",
    "or_duration": "OR duration",
    "surgery_duration": "Surgery duration",
    "anesthesia_duration": "Anesthesia duration",

    # Interaction terms
    "int_age_x_emergency": "Age × emergency",
    "int_asa_x_emergency": "ASA × emergency",
    "int_age_x_or_duration": "Age × OR duration",
    "int_asa_x_or_duration": "ASA × OR duration",
    "int_age_x_surgery_duration": "Age × surgery duration",
    "int_asa_x_surgery_duration": "ASA × surgery duration",
    "int_emergency_x_or_duration": "Emergency × OR duration",
    "int_emergency_x_surgery_duration": "Emergency × surgery duration",
    "int_emergency_x_anesthesia_duration": "Emergency × anesthesia duration",

    # Preoperative laboratory values
    "preop_fibrinogen": "Preop fibrinogen",
    "preop_platelet": "Preop platelets",
    "preop_alp": "Preop ALP",
    "preop_glucose": "Preop glucose",
    "preop_lymphocyte": "Preop lymphocytes",
    "preop_ast": "Preop AST",
    "preop_alt": "Preop ALT",
    "preop_albumin": "Preop albumin",
    "preop_creatinine": "Preop creatinine",
    "preop_hb": "Preop hemoglobin",
    "preop_hct": "Preop hematocrit",
    "preop_wbc": "Preop WBC",
    "preop_sodium": "Preop sodium",
    "preop_potassium": "Preop potassium",
    "preop_ptinr": "Preop coagulation (PT-INR)",
    "preop_mean_hr": "Preop mean HR",
    "preop_mean_rr": "Preop mean RR",

    # Intraoperative variables
    "intraop_mean_ebl": "Intraop mean blood loss",
    "intraop_max_ebl": "Intraop max blood loss",
    "intraop_min_ebl": "Intraop min blood loss",
    "intraop_delta_ebl": "Intraop Δ blood loss",
    "intraop_mean_abs_delta_ebl": "Intraop mean Δ blood loss",
    # INSPIRE psa = infused plasma solution volume, not prostate-specific antigen.
    "intraop_mean_psa": "Intraop mean plasma solution",
    "intraop_max_psa": "Intraop max plasma solution",
    "intraop_min_psa": "Intraop min plasma solution",
    "intraop_delta_psa": "Intraop Δ plasma solution",
    "intraop_mean_abs_delta_psa": "Intraop mean Δ plasma solution",
    "intraop_mean_hr": "Intraop mean HR",
    "intraop_max_hr": "Intraop max HR",
    "intraop_mean_art_dbp": "Intraop mean ART DBP",
    "intraop_mean_art_mbp": "Intraop mean ART MBP",
    "intraop_mean_art_sbp": "Intraop mean ART SBP",
    "intraop_mean_etco2": "Intraop mean EtCO₂",
    "intraop_std_spo2": "Intraop SpO₂ variability",

    # Diagnoses
    "diag_neoplasms": "Neoplasm diagnosis",
    "diag_eyeear": "Eye/ear diagnosis",
    "diag_circulatory": "Circulatory diagnosis",
    "diag_digestive": "Digestive diagnosis",
    "diag_respiratory": "Respiratory diagnosis",
    "diag_endocrine": "Endocrine diagnosis",
    "diag_genitourinary": "Genitourinary diagnosis",
    "diag_musculoskeletal": "Musculoskeletal diagnosis",
    "diag_infectious": "Infectious diagnosis",

    # Medication categories
    "preop_med_a02": "Preop acid-related med",
    "preop_med_a06": "Preop laxative",
    "preop_med_j01": "Preop antibiotic",
    "preop_med_nan": "Preop med missing",
    "preop_med_missing": "Preop med missing",
    "preop_med_unknown": "Preop med missing",
}


SPECIAL_UPPER = {
    "hr", "rr", "spo2", "fio2", "uo", "dbp", "mbp", "sbp", "bp",
    "vt", "pip", "peep", "pplat", "pmean", "hct", "hb", "nibp",
    "abp", "art", "etco2", "alp", "ast", "alt", "cci", "ptinr",
    "wbc", "bmi", "asa", "or",
}


CLINICAL_TERM_MAP = {
    "ebl": "blood loss",
    "psa": "plasma solution",
    "platelet": "platelets",
    "lymphocyte": "lymphocytes",
    "alp": "ALP",
    "hb": "hemoglobin",
    "hct": "hematocrit",
    "wbc": "WBC",
    "ptinr": "coagulation (PT-INR)",
    "art_dbp": "ART DBP",
    "art_mbp": "ART MBP",
    "art_sbp": "ART SBP",
    "etco2": "EtCO₂",
    "spo2": "SpO₂",
    "fio2": "FiO₂",
}


DIAGNOSIS_TERM_MAP = {
    "eyeear": "Eye/ear diagnosis",
    "neoplasms": "Neoplasm diagnosis",
    "circulatory": "Circulatory diagnosis",
    "digestive": "Digestive diagnosis",
    "respiratory": "Respiratory diagnosis",
    "endocrine": "Endocrine diagnosis",
    "genitourinary": "Genitourinary diagnosis",
    "musculoskeletal": "Musculoskeletal diagnosis",
    "infectious": "Infectious diagnosis",
}


def pretty_feat(name: str) -> str:
    """Convert transformed feature names into short, tidy figure labels."""
    n = str(name)
    n = re.sub(r"^(num|cat)__", "", n)

    is_missing = n.startswith("missingindicator_")
    if is_missing:
        n = n[len("missingindicator_"):]

    mapped_label = SHORT_LABEL_MAP.get(n.lower())
    if mapped_label is not None:
        return f"Missing: {mapped_label}" if is_missing else mapped_label

    if n.lower().startswith("int_"):
        label = (
            n[4:]
            .replace("_x_", " × ")
            .replace("_", " ")
            .lower()
        )
        label = label[0].upper() + label[1:] if label else "Interaction"
        label = label.replace(" or ", " OR ")
        return f"Missing: {label}" if is_missing else label

    if n.upper().startswith("DIAG_"):
        core = n[5:].strip("_")
        label = DIAGNOSIS_TERM_MAP.get(
            core.lower(),
            core.replace("_", " ").capitalize() + " diagnosis",
        )
        return f"Missing: {label}" if is_missing else label

    if n.upper().startswith("CCI_"):
        core = n[4:].replace("_", " ").strip()
        label = f"CCI: {core.capitalize()}"
        return f"Missing: {label}" if is_missing else compress_words(label)

    if n.upper().startswith("PROC_"):
        core = n[5:].replace("_", " ").strip()
        label = f"Procedure: {core.capitalize()}"
        return f"Missing: {label}" if is_missing else compress_words(label)

    if n.lower().startswith("preop_med_"):
        core = n[len("preop_med_"):].strip("_")
        medication_label_map = {
            "a02": "Preop acid-related med",
            "a06": "Preop laxative",
            "j01": "Preop antibiotic",
        }
        if core.lower() in {"nan", "na", "none", "missing", "unknown"}:
            label = "Preop med missing"
        elif core.lower() in medication_label_map:
            label = medication_label_map[core.lower()]
        else:
            label = f"Preop med {core.upper()}"
        return f"Missing: {label}" if is_missing else compress_words(label)

    phase = ""
    for prefix, phase_name in [
        ("preop_", "Preop "),
        ("postop_", "Postop "),
        ("intraop_", "Intraop "),
    ]:
        if n.lower().startswith(prefix):
            phase = phase_name
            n = n[len(prefix):]
            break

    metric = ""
    if n.lower().startswith("mean_abs_delta_"):
        metric = "Mean Δ "
        n = n[len("mean_abs_delta_"):]
    else:
        match = re.match(
            r"^(mean|min|max|std|range|auc|trend|delta|median)_",
            n,
            flags=re.IGNORECASE,
        )
        if match:
            key = match.group(1).lower()
            metric = {
                "mean": "Mean ",
                "min": "Min ",
                "max": "Max ",
                "std": "Variability in ",
                "range": "Range of ",
                "auc": "Cumulative ",
                "trend": "Trend in ",
                "delta": "Δ ",
                "median": "Median ",
            }[key]
            n = n[len(match.group(1)) + 1:]

    clinical_term = CLINICAL_TERM_MAP.get(n.lower())
    if clinical_term is not None:
        label = (phase + metric + clinical_term).strip() if metric else (phase + clinical_term).strip()
    else:
        tokens = []
        for token in n.split("_"):
            lower = token.lower()
            if lower in CLINICAL_TERM_MAP:
                tokens.append(CLINICAL_TERM_MAP[lower])
            elif lower in SPECIAL_UPPER:
                formatted = lower.upper()
                formatted = formatted.replace("SPO2", "SpO₂")
                formatted = formatted.replace("FIO2", "FiO₂")
                formatted = formatted.replace("ETCO2", "EtCO₂")
                formatted = formatted.replace("PTINR", "coagulation (PT-INR)")
                tokens.append(formatted)
            elif len(token) <= 3:
                tokens.append(token.upper())
            else:
                tokens.append(token.lower())
        core_label = " ".join(tokens).strip()
        label = (phase + metric + core_label).strip() if metric else (phase + core_label).strip()

    label = re.sub(r"\s+", " ", label).strip()
    if label:
        label = label[0].upper() + label[1:]
    if is_missing:
        label = "Missing: " + label
    return compress_words(label, front=5, back=3)


def compact_raw_feature_name(name: str) -> str:
    """Create a concise readable suffix from the exact saved transformed feature name."""
    raw = str(name).strip()
    raw = re.sub(r"^(num|cat)__", "", raw)
    is_missing = raw.startswith("missingindicator_")
    if is_missing:
        raw = raw[len("missingindicator_"):]

    readable = pretty_feat(raw)
    readable = readable.replace("Missing: ", "")
    qualifiers = []
    lower = raw.lower()
    if is_missing:
        qualifiers.append("missing")
    if lower.startswith("preop_"):
        qualifiers.append("preop")
    elif lower.startswith("intraop_"):
        qualifiers.append("intraop")
    elif lower.startswith("postop_"):
        qualifiers.append("postop")
    elif lower.startswith("diag_"):
        qualifiers.append("diagnosis")
    elif lower.startswith("proc_"):
        qualifiers.append("procedure")

    qualifier = ", ".join(dict.fromkeys(q for q in qualifiers if q))
    if qualifier and qualifier.lower() not in readable.lower():
        return f"{readable} | {qualifier}"
    return readable


def compact_label_qualifier(name: str) -> str:
    """Short qualifier used only when multiple features share one base label."""
    raw = re.sub(r"^(num|cat)__", "", str(name).strip())
    is_missing = raw.startswith("missingindicator_")
    if is_missing:
        raw = raw[len("missingindicator_"):]

    lower = raw.lower()
    qualifiers = []
    if lower.startswith("preop_"):
        qualifiers.append("preop")
    elif lower.startswith("intraop_"):
        qualifiers.append("intraop")
    elif lower.startswith("postop_"):
        qualifiers.append("postop")
    elif lower.startswith("diag_"):
        qualifiers.append("diagnosis")
    elif lower.startswith("proc_"):
        qualifiers.append("procedure")

    if is_missing:
        qualifiers.append("missing")

    qualifiers = list(dict.fromkeys([q for q in qualifiers if q]))
    return ", ".join(qualifiers)


def make_unique_feature_labels(
    feature_names: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    """Generate concise labels that remain unique within one component."""
    raw_names = [str(name) for name in feature_names]
    base_labels = [pretty_feat(name) for name in raw_names]
    base_counts = Counter(base_labels)

    candidate_labels = []
    for raw_name, base_label in zip(raw_names, base_labels):
        if base_counts[base_label] == 1:
            candidate = base_label
        else:
            qualifier = compact_label_qualifier(raw_name)
            candidate = (
                f"{base_label} ({qualifier})"
                if qualifier
                else base_label
            )
        candidate_labels.append(candidate)

    candidate_counts = Counter(candidate_labels)
    fallback_labels = []
    for raw_name, base_label, candidate in zip(raw_names, base_labels, candidate_labels):
        if candidate_counts[candidate] == 1:
            fallback_labels.append(candidate)
        else:
            fallback_labels.append(f"{base_label} [{compact_raw_feature_name(raw_name)}]")

    seen = Counter()
    final_labels = []
    for label in fallback_labels:
        seen[label] += 1
        occurrence = seen[label]
        final_labels.append(label if occurrence == 1 else f"{label} #{occurrence}")

    audit = pd.DataFrame(
        {
            "feature": raw_names,
            "base_pretty_label": base_labels,
            "candidate_pretty_label": candidate_labels,
            "final_pretty_label": final_labels,
            "base_label_was_duplicated": [base_counts[label] > 1 for label in base_labels],
        }
    )

    if pd.Series(final_labels).duplicated().any():
        raise RuntimeError(
            "Feature-label disambiguation failed; duplicate display labels remain."
        )

    return final_labels, audit


def assert_unique_display_labels(
    feature_table: pd.DataFrame,
    *,
    top_k: int,
    component_name: str,
) -> None:
    """Stop figure generation if displayed labels are still duplicated."""
    displayed = (
        feature_table.head(top_k)["pretty_feature"]
        .astype(str)
    )
    duplicated = displayed.duplicated(keep=False)

    if duplicated.any():
        duplicate_rows = feature_table.head(top_k).loc[
            duplicated,
            ["feature", "pretty_feature"],
        ]
        raise RuntimeError(
            f"{component_name} still contains duplicated display labels:\n"
            + duplicate_rows.to_string(index=False)
        )


# =============================================================================
# CURRENT-RUN RECONSTRUCTION
# =============================================================================

def reconstruct_current_run(
    module,
    run_dir: pathlib.Path,
    data_path: pathlib.Path,
) -> Dict[str, Any]:
    config = load_json(run_dir / "config.json")

    configured_data_path = pathlib.Path(str(config.get("data_path", data_path)))
    if data_path == DEFAULT_DATA_PATH and configured_data_path.exists():
        data_path = configured_data_path

    print("\n" + "=" * 100)
    print("RECONSTRUCTING FINAL TEMPORAL-SAFE COHORT AND SAVED SPLIT ORDER")
    print("=" * 100)

    df_input = pd.read_csv(require_file(data_path), low_memory=False)
    df, cohort_flow = module.apply_reviewer_cohort_exclusions(
        df_input,
        target_column=str(config.get("target_column", "mortality_30d")),
        exclude_asa6=bool(config.get("exclude_asa6", True)),
    )

    if bool(config.get("add_interactions", True)):
        df = module.add_mortality_interactions(df)

    X_raw = module.extract_model_features_classification(df, verbose=False)
    target_column = str(config.get("target_column", "mortality_30d"))
    y_all = df.loc[X_raw.index, target_column].astype(int).to_numpy()
    X_raw = module.clean_raw_predictors(X_raw)

    test_predictions = pd.read_csv(
        require_file(run_dir / "test_predictions_all_models.csv")
    )
    development_predictions = pd.read_csv(
        require_file(run_dir / "development_oof_predictions.csv")
    )

    required_test_columns = {
        "final_cohort_row_position",
        "source_row_position",
        "y_true",
    }
    required_dev_columns = {
        "final_cohort_row_position",
        "source_row_position",
        "y_true",
    }
    if not required_test_columns.issubset(test_predictions.columns):
        raise KeyError(
            "test_predictions_all_models.csv lacks required columns: "
            f"{sorted(required_test_columns - set(test_predictions.columns))}"
        )
    if not required_dev_columns.issubset(development_predictions.columns):
        raise KeyError(
            "development_oof_predictions.csv lacks required columns: "
            f"{sorted(required_dev_columns - set(development_predictions.columns))}"
        )

    test_positions = test_predictions[
        "final_cohort_row_position"
    ].astype(int).to_numpy()
    development_positions = development_predictions[
        "final_cohort_row_position"
    ].astype(int).to_numpy()

    X_test_raw = X_raw.iloc[test_positions].reset_index(drop=True)
    X_development_raw = X_raw.iloc[development_positions].reset_index(drop=True)

    y_test = y_all[test_positions]
    y_development = y_all[development_positions]

    if not np.array_equal(
        y_test,
        test_predictions["y_true"].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "Reconstructed test outcomes do not match saved test predictions."
        )
    if not np.array_equal(
        y_development,
        development_predictions["y_true"].astype(int).to_numpy(),
    ):
        raise RuntimeError(
            "Reconstructed development outcomes do not match saved OOF predictions."
        )

    print(cohort_flow.to_string(index=False))
    print(f"Development encounters: {len(y_development):,}")
    print(f"Development deaths:     {int(y_development.sum()):,}")
    print(f"Test encounters:        {len(y_test):,}")
    print(f"Test deaths:            {int(y_test.sum()):,}")
    print(f"Raw predictors:         {X_raw.shape[1]:,}")

    return {
        "config": config,
        "data_path": data_path,
        "X_development_raw": X_development_raw,
        "X_test_raw": X_test_raw,
        "y_development": y_development,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "development_predictions": development_predictions,
        "test_positions": test_positions,
    }


def select_attribution_sample(
    y_test: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    positive = np.where(y_test == 1)[0]
    negative = np.where(y_test == 0)[0]

    remaining = max(0, max_samples - len(positive))
    selected_negative = rng.choice(
        negative,
        size=min(remaining, len(negative)),
        replace=False,
    )
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    return selected.astype(int)


# =============================================================================
# DIRECT GRADIENTSHAP
# =============================================================================

def compute_gradient_shap_batches(
    model: torch.nn.Module,
    X: np.ndarray,
    baselines: np.ndarray,
    *,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    try:
        from captum.attr import GradientShap
    except Exception as exc:
        raise ImportError(
            "Captum is required for direct GradientSHAP. "
            "Install it in the active environment with `pip install captum`."
        ) from exc

    model.eval()
    explainer = GradientShap(model)

    baseline_tensor = torch.as_tensor(
        np.asarray(baselines, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    outputs: List[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        batch = torch.as_tensor(
            np.asarray(X[start:stop], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        attribution = explainer.attribute(
            batch,
            baselines=baseline_tensor,
            n_samples=n_samples,
            stdevs=0.0,
        )
        outputs.append(attribution.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float64)


def transformed_matrix_for_fold(
    raw_frame: pd.DataFrame,
    preprocessor,
    variance_selector,
    selected_indices: np.ndarray,
) -> np.ndarray:
    matrix = preprocessor.transform(raw_frame)
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    matrix = variance_selector.transform(matrix).astype(np.float32)
    return matrix[:, selected_indices].astype(np.float32)


def feature_domain_map_from_metadata(
    feature_names: np.ndarray,
    metadata: Mapping[str, Any],
) -> Dict[str, str]:
    output: Dict[str, str] = {}
    domain_indices = metadata.get("domain_indices", {})
    for domain, indices in domain_indices.items():
        for index in indices:
            index = int(index)
            if 0 <= index < len(feature_names):
                output[str(feature_names[index])] = str(domain)
    return output


def compute_current_run_attributions(
    module,
    reconstructed: Mapping[str, Any],
    run_dir: pathlib.Path,
    sample_indices: np.ndarray,
    *,
    background_n: int,
    gradient_samples: int,
    attribution_batch_size: int,
    device: torch.device,
    cache_path: pathlib.Path,
    force_recompute: bool,
) -> Dict[str, Any]:
    if cache_path.exists() and not force_recompute:
        print("\nLoading cached strict-OOF ensemble attributions:", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        cached_sample = cached["sample_indices"].astype(int)

        if not np.array_equal(cached_sample, sample_indices):
            raise RuntimeError(
                "The cached Uniform DARN attribution sample differs from the "
                "current requested test sample. Re-run with "
                "--force_recompute."
            )

        return {
            "features": normalize_object_array(cached["features"]),
            "domains": normalize_object_array(cached["domains"]),
            "attributions": cached["attributions"].astype(np.float64),
            "feature_values": cached["feature_values"].astype(np.float64),
            "presence_count": cached["presence_count"].astype(int),
            "sample_indices": cached_sample,
        }

    config = reconstructed["config"]
    X_development_raw = reconstructed["X_development_raw"]
    X_test_raw = reconstructed["X_test_raw"]
    y_development = reconstructed["y_development"]

    n_folds = int(config.get("n_folds", 5))
    random_state = int(config.get("random_state", 42))

    cross_validator = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    fold_splits = list(
        cross_validator.split(X_development_raw, y_development)
    )

    fold_results: List[Dict[str, Any]] = []
    domain_votes: Dict[str, Counter] = defaultdict(Counter)

    print("\n" + "=" * 100)
    print("DIRECT GRADIENTSHAP ACROSS SAVED UNIFORM-DARN FOLDS")
    print("=" * 100)
    print(f"Device:              {device}")
    print(f"Attribution sample:  {len(sample_indices):,}")
    print(f"Gradient samples:    {gradient_samples}")
    print(f"Background per fold: {background_n}")

    for fold_number, (fold_train_indices, _) in enumerate(
        fold_splits,
        start=1,
    ):
        fold_dir = run_dir / "fold_artifacts" / f"fold_{fold_number}"
        print("\n" + "-" * 100)
        print(f"FOLD {fold_number}/{n_folds}")
        print("-" * 100)

        preprocessor = joblib.load(
            require_file(fold_dir / "preprocessor.joblib")
        )
        variance_selector = joblib.load(
            require_file(fold_dir / "variance_selector.joblib")
        )
        selected_indices = np.load(
            require_file(fold_dir / "selected_original_indices.npy")
        ).astype(int)
        selected_feature_names = normalize_object_array(
            np.load(
                require_file(fold_dir / "selected_feature_names.npy"),
                allow_pickle=True,
            )
        )
        metadata = load_json(
            fold_dir / "selected_domain_metadata.json"
        )
        domain_names = list(metadata["domain_names"])
        domain_slices = [
            np.asarray(metadata["domain_indices"][domain], dtype=np.int64)
            for domain in domain_names
        ]

        feature_domain_map = feature_domain_map_from_metadata(
            selected_feature_names,
            metadata,
        )
        for feature, domain in feature_domain_map.items():
            domain_votes[feature][domain] += 1

        X_sample = transformed_matrix_for_fold(
            X_test_raw.iloc[sample_indices].copy(),
            preprocessor,
            variance_selector,
            selected_indices,
        )

        rng = np.random.default_rng(random_state + fold_number * 1000)
        selected_background_positions = rng.choice(
            fold_train_indices,
            size=min(background_n, len(fold_train_indices)),
            replace=False,
        )
        X_background = transformed_matrix_for_fold(
            X_development_raw.iloc[selected_background_positions].copy(),
            preprocessor,
            variance_selector,
            selected_indices,
        )

        state_paths = sorted(fold_dir.glob("uniform_darn_seed_*.pt"))
        if not state_paths:
            raise FileNotFoundError(
                f"No Uniform DARN state files found in {fold_dir}"
            )

        seed_attributions: List[np.ndarray] = []
        for state_path in state_paths:
            print("Explaining:", state_path.name)
            model = module.UniformDARNClassifier(
                domain_slices=domain_slices,
                hidden_dim=int(config.get("darn_hidden_dim", 48)),
                dropout=float(config.get("darn_dropout", 0.30)),
                feature_dropout=float(
                    config.get("darn_feature_dropout", 0.10)
                ),
            ).to(device)

            state_dict = safe_torch_load(state_path, device)
            model.load_state_dict(state_dict)
            model.eval()

            seed_attributions.append(
                compute_gradient_shap_batches(
                    model,
                    X_sample,
                    X_background,
                    n_samples=gradient_samples,
                    batch_size=attribution_batch_size,
                    device=device,
                )
            )

            model.to("cpu")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_attribution = np.mean(
            np.stack(seed_attributions, axis=0),
            axis=0,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "features": selected_feature_names,
                "attributions": fold_attribution,
                "feature_values": X_sample.astype(np.float64),
            }
        )

        np.savez_compressed(
            cache_path.parent / f"fold_{fold_number}_direct_gradientshap.npz",
            features=selected_feature_names,
            sample_indices=sample_indices,
            attributions=fold_attribution,
            feature_values=X_sample.astype(np.float64),
        )

        print(
            f"Fold features: {len(selected_feature_names):,} | "
            f"Attribution matrix: {fold_attribution.shape}"
        )

    union_features = sorted(
        {
            str(feature)
            for fold_result in fold_results
            for feature in fold_result["features"]
        }
    )
    feature_to_union = {
        feature: index
        for index, feature in enumerate(union_features)
    }

    n_sample = len(sample_indices)
    n_union = len(union_features)
    attribution_sum = np.zeros((n_sample, n_union), dtype=np.float64)
    value_sum = np.zeros((n_sample, n_union), dtype=np.float64)
    value_count = np.zeros((n_sample, n_union), dtype=np.float64)
    presence_count = np.zeros(n_union, dtype=int)

    for fold_result in fold_results:
        features = fold_result["features"]
        indices = np.asarray(
            [feature_to_union[str(feature)] for feature in features],
            dtype=int,
        )
        attribution_sum[:, indices] += fold_result["attributions"]
        value_sum[:, indices] += fold_result["feature_values"]
        value_count[:, indices] += 1.0
        presence_count[indices] += 1

    # The saved Uniform DARN prediction is an average across folds. A feature
    # absent from a fold has zero contribution in that fold, so signed
    # attributions are averaged across all folds, not only folds containing it.
    ensemble_attributions = attribution_sum / float(len(fold_results))

    ensemble_feature_values = np.divide(
        value_sum,
        value_count,
        out=np.zeros_like(value_sum),        where=value_count > 0,
    )

    final_domains: List[str] = []
    for feature in union_features:
        votes = domain_votes.get(feature, Counter())
        final_domains.append(
            votes.most_common(1)[0][0] if votes else "Other"
        )

    result = {
        "features": np.asarray(union_features, dtype=object),
        "domains": np.asarray(final_domains, dtype=object),
        "attributions": ensemble_attributions,
        "feature_values": ensemble_feature_values,
        "presence_count": presence_count,
        "sample_indices": sample_indices,
    }

    np.savez_compressed(
        cache_path,
        features=result["features"],
        domains=result["domains"],
        attributions=result["attributions"],
        feature_values=result["feature_values"],
        presence_count=result["presence_count"],
        sample_indices=result["sample_indices"],
    )

    print("\nSaved ensemble attribution cache:", cache_path)
    print(f"Union transformed features: {n_union:,}")
    return result


# =============================================================================
# TABLES
# =============================================================================

def make_importance_tables(
    features: np.ndarray,
    domains: np.ndarray,
    attribution_values: np.ndarray,
    presence_count: np.ndarray,
    n_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_labels, _ = make_unique_feature_labels(
        features.astype(str)
    )

    feature_names = features.astype(str)
    missing_indicator_mask = np.asarray(
        [
            bool(
                re.sub(r"^(num|cat)__", "", feature_name)
                .startswith("missingindicator_")
            )
            for feature_name in feature_names
        ],
        dtype=bool,
    )

    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "pretty_feature": unique_labels,
            "domain": domains.astype(str),
            "is_missing_indicator": missing_indicator_mask,
            "mean_absolute_direct_attribution": np.mean(
                np.abs(attribution_values),
                axis=0,
            ),
            "mean_signed_direct_attribution": np.mean(
                attribution_values,
                axis=0,
            ),
            "median_signed_direct_attribution": np.median(
                attribution_values,
                axis=0,
            ),
            "proportion_positive": np.mean(
                attribution_values > 0,
                axis=0,
            ),
            "proportion_negative": np.mean(
                attribution_values < 0,
                axis=0,
            ),
            "fold_presence_count": presence_count.astype(int),
            "fold_presence_fraction": presence_count / max(n_folds, 1),
        }
    ).sort_values(
        "mean_absolute_direct_attribution",
        ascending=False,
    ).reset_index(drop=True)

    feature_table.insert(
        0,
        "rank",
        np.arange(1, len(feature_table) + 1),
    )

    domain_rows: List[Dict[str, Any]] = []
    for domain in sorted(set(domains.astype(str))):
        indices = np.where(domains.astype(str) == domain)[0]
        patient_domain_attribution = attribution_values[:, indices].sum(axis=1)
        domain_rows.append(
            {
                "domain": domain,
                "n_union_features": int(len(indices)),
                "mean_absolute_domain_attribution": float(
                    np.mean(np.abs(patient_domain_attribution))
                ),
                "mean_signed_domain_attribution": float(
                    np.mean(patient_domain_attribution)
                ),
                "median_signed_domain_attribution": float(
                    np.median(patient_domain_attribution)
                ),
            }
        )

    domain_table = pd.DataFrame(domain_rows).sort_values(
        "mean_absolute_domain_attribution",
        ascending=False,
    ).reset_index(drop=True)
    domain_table.insert(
        0,
        "rank",
        np.arange(1, len(domain_table) + 1),
    )

    return feature_table, domain_table


# =============================================================================
# PLOTS
# =============================================================================

def feature_table_for_display(
    feature_table: pd.DataFrame,
    *,
    exclude_missing_indicators: bool = EXCLUDE_MISSING_INDICATORS_FROM_DISPLAY,
) -> pd.DataFrame:
    """
    Return the manuscript-facing feature ranking.

    The complete attribution table remains unchanged and retains all
    missingness indicators. Only the displayed beeswarm and console ranking
    omit them.
    """
    table = feature_table.copy()

    if exclude_missing_indicators:
        if "is_missing_indicator" not in table.columns:
            table["is_missing_indicator"] = (
                table["feature"]
                .astype(str)
                .str.replace(
                    r"^(num|cat)__",
                    "",
                    regex=True,
                )
                .str.startswith("missingindicator_")
            )

        table = table.loc[
            ~table["is_missing_indicator"].astype(bool)
        ].copy()

    return (
        table.sort_values(
            "mean_absolute_direct_attribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def configure_style() -> None:
    """Use the same publication typography as Figures 2 and 3."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "figure.titlesize": 18,
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
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight")
    try:
        fig.savefig(
            png_path.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception as exc:
        print("TIFF export skipped:", exc)


def plot_beeswarm(
    attribution_values: np.ndarray,
    feature_values: np.ndarray,
    feature_table: pd.DataFrame,
    features: np.ndarray,
    output_path: pathlib.Path,
    top_k: int,
    random_state: int,
) -> None:
    top_features = feature_table.head(top_k)["feature"].astype(str).tolist()
    feature_to_index = {
        str(feature): index
        for index, feature in enumerate(features)
    }
    top_indices_desc = np.asarray(
        [feature_to_index[feature] for feature in top_features],
        dtype=int,
    )
    top_indices = top_indices_desc[::-1]

    attribution_top = attribution_values[:, top_indices]
    values_top = feature_values[:, top_indices]
    label_lookup = (
        feature_table.set_index("feature")["pretty_feature"]
        .astype(str)
        .to_dict()
    )
    labels = [
        label_lookup[str(features[index])]
        for index in top_indices
    ]

    feature_min = np.nanmin(values_top, axis=0)
    feature_max = np.nanmax(values_top, axis=0)
    denominator = np.where(
        feature_max - feature_min == 0,
        1.0,
        feature_max - feature_min,
    )
    values_scaled = (values_top - feature_min) / denominator

    cmap = cm.get_cmap(CMAP_NAME)
    rng = np.random.default_rng(random_state)

    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(1, 24)
    ax = fig.add_subplot(grid[0, :23])
    cax = fig.add_subplot(grid[0, 23])

    ax.axvline(0, color="dimgray", linewidth=1.1, alpha=0.8)

    for feature_position in range(len(labels)):
        x = attribution_top[:, feature_position]
        y = (
            np.full(len(x), feature_position, dtype=float)
            + (rng.random(len(x)) - 0.5) * BEESWARM_JITTER
        )
        colors = cmap(
            np.clip(values_scaled[:, feature_position], 0.0, 1.0)
        )
        ax.scatter(
            x,
            y,
            s=BEESWARM_DOT_SIZE,
            c=colors,
            edgecolor="none",
            alpha=BEESWARM_ALPHA,
            rasterized=True,
        )

        median = float(np.median(x))
        ax.plot(
            [median, median],
            [feature_position - 0.34, feature_position + 0.34],
            color="black",
            linewidth=1.8,
            alpha=0.35,
        )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xlabel(
        "Direct Uniform DARN attribution (contribution to mortality logit)"
    )
    ax.set_title(
        "A. Feature-level contributions to Uniform DARN predictions",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.22)

    colorbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label(
        "Transformed feature value",
        rotation=270,
        labelpad=16,
    )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


def plot_global_importance(
    feature_table: pd.DataFrame,
    output_path: pathlib.Path,
    top_k: int,
) -> None:
    table = (
        feature_table.head(top_k)
        .sort_values("mean_absolute_direct_attribution", ascending=True)
        .reset_index(drop=True)
    )
    cmap = cm.get_cmap(CMAP_NAME)
    colors = [
        cmap(index / max(len(table) - 1, 1))
        for index in range(len(table))
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(
        np.arange(len(table)),
        table["mean_absolute_direct_attribution"],
        color=colors,
        edgecolor="none",
    )
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["pretty_feature"])
    ax.set_xlabel("Mean absolute direct Uniform DARN attribution")
    ax.set_title(
        "B. Global feature importance in the Uniform DARN component",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    maximum = max(
        float(table["mean_absolute_direct_attribution"].max()),
        1e-12,
    )
    ax.set_xlim(0, maximum * 1.18)
    for position, value in enumerate(
        table["mean_absolute_direct_attribution"]
    ):
        ax.text(
            value + maximum * 0.012,
            position,
            f"{value:.3f}",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


def plot_patient_attribution(
    attribution_values: np.ndarray,
    features: np.ndarray,
    sample_indices: np.ndarray,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    test_predictions: pd.DataFrame,
    output_path: pathlib.Path,
    patient_top_k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sample_y = y_test[sample_indices]
    sample_probability = probabilities[sample_indices]

    event_local_indices = np.where(sample_y == 1)[0]
    if len(event_local_indices):
        patient_local_index = int(
            event_local_indices[
                np.argmax(sample_probability[event_local_indices])
            ]
        )
    else:
        patient_local_index = int(np.argmax(sample_probability))

    patient_attribution = attribution_values[patient_local_index]
    top_indices = np.argsort(
        np.abs(patient_attribution)
    )[::-1][:patient_top_k]

    patient_features = features[top_indices].astype(str)
    patient_labels, _ = make_unique_feature_labels(
        patient_features
    )

    patient_table = pd.DataFrame(
        {
            "feature": patient_features,
            "pretty_feature": patient_labels,
            "signed_attribution": patient_attribution[top_indices],
            "absolute_attribution": np.abs(
                patient_attribution[top_indices]
            ),
        }
    ).sort_values(
        "signed_attribution",
        ascending=True,
    ).reset_index(drop=True)

    test_row_position = int(sample_indices[patient_local_index])
    final_cohort_position = int(
        test_predictions.iloc[test_row_position][
            "final_cohort_row_position"
        ]
    )
    source_row_position = int(
        test_predictions.iloc[test_row_position]["source_row_position"]
    )
    outcome = int(y_test[test_row_position])
    probability = float(probabilities[test_row_position])

    cmap = cm.get_cmap(CMAP_NAME)
    positive_color = cmap(0.82)
    negative_color = cmap(0.18)

    contributions = patient_table["signed_attribution"].to_numpy(float)
    maximum = max(float(np.max(np.abs(contributions))), 1e-12)

    fig, ax = plt.subplots(figsize=(13, 9))
    for position, contribution in enumerate(contributions):
        left = contribution if contribution < 0 else 0
        width = abs(contribution)
        color = positive_color if contribution >= 0 else negative_color
        ax.barh(
            position,
            width,
            left=left,
            height=0.64,
            color=color,
            edgecolor="none",
        )
        offset = 0.025 * maximum
        ax.text(
            contribution + (offset if contribution >= 0 else -offset),
            position,
            f"{contribution:+.3f}",
            va="center",
            ha="left" if contribution >= 0 else "right",
            fontsize=11,
        )

    ax.axvline(0, color="dimgray", linewidth=1.1)
    ax.set_yticks(np.arange(len(patient_table)))
    ax.set_yticklabels(patient_table["pretty_feature"])
    ax.set_xlim(-1.25 * maximum, 1.25 * maximum)
    ax.set_xlabel(
        "Signed direct Uniform DARN attribution (mortality-logit contribution)"
    )
    ax.set_title(
        (
            "C. Patient-level attribution for a high-risk death\n"
            f"Observed outcome={outcome}; calibrated DARN risk={probability:.4f}"
        ),
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                color=positive_color,
                linewidth=7,
                label="Increases predicted mortality",
            ),
            Line2D(
                [0], [0],
                color=negative_color,
                linewidth=7,
                label="Decreases predicted mortality",
            ),
        ],
        loc="lower right",
    )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)

    metadata = {
        "attribution_sample_row": patient_local_index,
        "test_row_position": test_row_position,
        "final_cohort_row_position": final_cohort_position,
        "source_row_position": source_row_position,
        "observed_outcome": outcome,
        "uniform_darn_calibrated_probability": probability,
    }
    return patient_table, metadata


def plot_domain_importance(
    domain_table: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    table = domain_table.sort_values(
        "mean_absolute_domain_attribution",
        ascending=True,
    ).reset_index(drop=True)
    cmap = cm.get_cmap(CMAP_NAME)
    colors = [
        cmap(0.15 + 0.75 * index / max(len(table) - 1, 1))
        for index in range(len(table))
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        np.arange(len(table)),
        table["mean_absolute_domain_attribution"],
        color=colors,
        edgecolor="none",
    )
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["domain"])
    ax.set_xlabel("Mean absolute summed feature attribution")
    ax.set_title(
        "D. Domain-level attribution in the Uniform DARN component",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    maximum = max(
        float(table["mean_absolute_domain_attribution"].max()),
        1e-12,
    )
    ax.set_xlim(0, maximum * 1.16)
    for position, value in enumerate(
        table["mean_absolute_domain_attribution"]
    ):
        ax.text(
            value + maximum * 0.012,
            position,
            f"{value:.3f}",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


# =============================================================================
# XGBOOST TREESHAP AND COMPONENT-LEVEL COMPARISON
# =============================================================================

def compute_xgboost_pred_contributions(
    model,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact XGBoost TreeSHAP contributions using Booster.predict with
    pred_contribs=True. Contributions are on the raw-margin scale.

    Returns
    -------
    feature_contributions
        Shape (n_samples, n_features).
    expected_margin
        Per-sample bias term returned by XGBoost.
    """
    try:
        import xgboost as xgb
    except Exception as exc:
        raise ImportError(
            "xgboost is required to explain the saved XGBoost models."
        ) from exc

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(
        np.asarray(X, dtype=np.float32),
    )

    predict_kwargs: Dict[str, Any] = {
        "pred_contribs": True,
        "validate_features": False,
    }

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        try:
            best_iteration_int = int(best_iteration)
            if best_iteration_int >= 0:
                predict_kwargs["iteration_range"] = (
                    0,
                    best_iteration_int + 1,
                )
        except Exception:
            pass

    try:
        contributions = booster.predict(
            dmatrix,
            **predict_kwargs,
        )
    except TypeError:
        # Compatibility fallback for older XGBoost versions.
        predict_kwargs.pop("iteration_range", None)
        best_ntree_limit = getattr(model, "best_ntree_limit", None)
        if best_ntree_limit is not None:
            try:
                predict_kwargs["ntree_limit"] = int(best_ntree_limit)
            except Exception:
                pass
        contributions = booster.predict(
            dmatrix,
            **predict_kwargs,
        )

    contributions = np.asarray(contributions)

    # Newer XGBoost releases can return (n, 1, p + 1) for binary models.
    if contributions.ndim == 3 and contributions.shape[1] == 1:
        contributions = contributions[:, 0, :]

    if contributions.ndim != 2:
        raise RuntimeError(
            "Unexpected XGBoost contribution shape: "
            f"{contributions.shape}"
        )

    expected_features = X.shape[1]
    if contributions.shape[1] != expected_features + 1:
        raise RuntimeError(
            "XGBoost TreeSHAP output does not match the selected feature "
            f"count: output={contributions.shape}, features={expected_features}."
        )

    return (
        contributions[:, :-1].astype(np.float64),
        contributions[:, -1].astype(np.float64),
    )


def compute_current_run_xgboost_attributions(
    reconstructed: Mapping[str, Any],
    run_dir: pathlib.Path,
    sample_indices: np.ndarray,
    *,
    cache_path: pathlib.Path,
    force_recompute: bool,
) -> Dict[str, Any]:
    """
    Explain the saved five-fold XGBoost ensemble on the same held-out test sample
    used for direct Uniform DARN attribution.

    Fold-specific TreeSHAP values are aligned by transformed feature name and
    averaged across all folds. A feature absent from a fold receives a zero
    contribution in that fold, matching the ensemble construction.
    """
    if cache_path.exists() and not force_recompute:
        print("\nLoading cached XGBoost TreeSHAP attributions:", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        cached_sample = cached["sample_indices"].astype(int)

        if not np.array_equal(cached_sample, sample_indices):
            raise RuntimeError(
                "The cached XGBoost attribution sample differs from the "
                "current Uniform DARN attribution sample. Re-run with "
                "--force_recompute."
            )

        return {
            "features": normalize_object_array(cached["features"]),
            "domains": normalize_object_array(cached["domains"]),
            "attributions": cached["attributions"].astype(np.float64),
            "feature_values": cached["feature_values"].astype(np.float64),
            "presence_count": cached["presence_count"].astype(int),
            "sample_indices": cached_sample,
            "mean_expected_margin": float(
                np.asarray(cached["mean_expected_margin"]).reshape(-1)[0]
            ),
        }

    config = reconstructed["config"]
    X_test_raw = reconstructed["X_test_raw"]
    n_folds = int(config.get("n_folds", 5))

    fold_results: List[Dict[str, Any]] = []
    domain_votes: Dict[str, Counter] = defaultdict(Counter)
    expected_margins: List[float] = []

    print("\n" + "=" * 100)
    print("DIRECT TREESHAP ACROSS SAVED XGBOOST FOLDS")
    print("=" * 100)
    print(f"Attribution sample: {len(sample_indices):,}")

    for fold_number in range(1, n_folds + 1):
        fold_dir = run_dir / "fold_artifacts" / f"fold_{fold_number}"

        print("\n" + "-" * 100)
        print(f"XGBOOST FOLD {fold_number}/{n_folds}")
        print("-" * 100)

        preprocessor = joblib.load(
            require_file(fold_dir / "preprocessor.joblib")
        )
        variance_selector = joblib.load(
            require_file(fold_dir / "variance_selector.joblib")
        )
        selected_indices = np.load(
            require_file(fold_dir / "selected_original_indices.npy")
        ).astype(int)
        selected_feature_names = normalize_object_array(
            np.load(
                require_file(fold_dir / "selected_feature_names.npy"),
                allow_pickle=True,
            )
        )
        metadata = load_json(
            fold_dir / "selected_domain_metadata.json"
        )
        xgb_model = joblib.load(
            require_file(fold_dir / "xgboost.joblib")
        )

        X_sample = transformed_matrix_for_fold(
            X_test_raw.iloc[sample_indices].copy(),
            preprocessor,
            variance_selector,
            selected_indices,
        )

        fold_attribution, expected_margin = (
            compute_xgboost_pred_contributions(
                xgb_model,
                X_sample,
            )
        )

        feature_domain_map = feature_domain_map_from_metadata(
            selected_feature_names,
            metadata,
        )
        for feature, domain in feature_domain_map.items():
            domain_votes[feature][domain] += 1

        expected_margins.append(float(np.mean(expected_margin)))

        fold_results.append(
            {
                "fold": fold_number,
                "features": selected_feature_names,
                "attributions": fold_attribution,
                "feature_values": X_sample.astype(np.float64),
            }
        )

        np.savez_compressed(
            cache_path.parent / f"fold_{fold_number}_xgboost_treeshap.npz",
            features=selected_feature_names,
            sample_indices=sample_indices,
            attributions=fold_attribution,
            feature_values=X_sample.astype(np.float64),
            expected_margin=expected_margin,
        )

        print(
            f"Fold features: {len(selected_feature_names):,} | "
            f"TreeSHAP matrix: {fold_attribution.shape}"
        )

    union_features = sorted(
        {
            str(feature)
            for fold_result in fold_results
            for feature in fold_result["features"]
        }
    )
    feature_to_union = {
        feature: index
        for index, feature in enumerate(union_features)
    }

    n_sample = len(sample_indices)
    n_union = len(union_features)
    attribution_sum = np.zeros((n_sample, n_union), dtype=np.float64)
    value_sum = np.zeros((n_sample, n_union), dtype=np.float64)
    value_count = np.zeros((n_sample, n_union), dtype=np.float64)
    presence_count = np.zeros(n_union, dtype=int)

    for fold_result in fold_results:
        indices = np.asarray(
            [
                feature_to_union[str(feature)]
                for feature in fold_result["features"]
            ],
            dtype=int,
        )
        attribution_sum[:, indices] += fold_result["attributions"]
        value_sum[:, indices] += fold_result["feature_values"]
        value_count[:, indices] += 1.0
        presence_count[indices] += 1

    ensemble_attributions = attribution_sum / float(n_folds)
    ensemble_feature_values = np.divide(
        value_sum,
        value_count,
        out=np.zeros_like(value_sum),
        where=value_count > 0,
    )

    final_domains: List[str] = []
    for feature in union_features:
        votes = domain_votes.get(feature, Counter())
        final_domains.append(
            votes.most_common(1)[0][0] if votes else "Other"
        )

    result = {
        "features": np.asarray(union_features, dtype=object),
        "domains": np.asarray(final_domains, dtype=object),
        "attributions": ensemble_attributions,
        "feature_values": ensemble_feature_values,
        "presence_count": presence_count,
        "sample_indices": sample_indices,
        "mean_expected_margin": float(np.mean(expected_margins)),
    }

    np.savez_compressed(
        cache_path,
        features=result["features"],
        domains=result["domains"],
        attributions=result["attributions"],
        feature_values=result["feature_values"],
        presence_count=result["presence_count"],
        sample_indices=result["sample_indices"],
        mean_expected_margin=np.asarray(
            [result["mean_expected_margin"]],
            dtype=float,
        ),
    )

    print("\nSaved XGBoost TreeSHAP cache:", cache_path)
    print(f"Union transformed features: {n_union:,}")
    return result


def normalized_feature_importance(
    feature_table: pd.DataFrame,
    value_column: str = "mean_absolute_direct_attribution",
) -> pd.DataFrame:
    table = feature_table.copy()
    total = max(float(table[value_column].sum()), 1e-15)
    table["normalized_importance"] = (
        table[value_column].astype(float) / total
    )
    table["importance_percent"] = (
        table["normalized_importance"] * 100.0
    )
    table["importance_rank"] = (
        table[value_column]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return table


def make_component_concordance_table(
    darn_feature_table: pd.DataFrame,
    xgb_feature_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    darn = normalized_feature_importance(
        darn_feature_table
    )[
        [
            "feature",
            "pretty_feature",
            "domain",
            "mean_absolute_direct_attribution",
            "normalized_importance",
            "importance_percent",
            "importance_rank",
        ]
    ].rename(
        columns={
            "domain": "darn_domain",
            "mean_absolute_direct_attribution": "darn_mean_abs_attribution",
            "normalized_importance": "darn_normalized_importance",
            "importance_percent": "darn_importance_percent",
            "importance_rank": "darn_rank",
        }
    )

    xgb = normalized_feature_importance(
        xgb_feature_table
    )[
        [
            "feature",
            "pretty_feature",
            "domain",
            "mean_absolute_direct_attribution",
            "normalized_importance",
            "importance_percent",
            "importance_rank",
        ]
    ].rename(
        columns={
            "pretty_feature": "xgb_pretty_feature",
            "domain": "xgb_domain",
            "mean_absolute_direct_attribution": "xgb_mean_abs_attribution",
            "normalized_importance": "xgb_normalized_importance",
            "importance_percent": "xgb_importance_percent",
            "importance_rank": "xgb_rank",
        }
    )

    merged = darn.merge(
        xgb,
        on="feature",
        how="outer",
    )

    merged["pretty_feature"] = (
        merged["pretty_feature"]
        .fillna(merged["xgb_pretty_feature"])
    )
    merged["domain"] = (
        merged["darn_domain"]
        .fillna(merged["xgb_domain"])
        .fillna("Other")
    )

    numeric_columns = [
        "darn_mean_abs_attribution",
        "darn_normalized_importance",
        "darn_importance_percent",
        "xgb_mean_abs_attribution",
        "xgb_normalized_importance",
        "xgb_importance_percent",
    ]
    for column in numeric_columns:
        merged[column] = merged[column].fillna(0.0)

    maximum_rank = len(merged) + 1
    merged["darn_rank"] = (
        merged["darn_rank"].fillna(maximum_rank).astype(int)
    )
    merged["xgb_rank"] = (
        merged["xgb_rank"].fillna(maximum_rank).astype(int)
    )

    merged["mean_normalized_importance"] = (
        merged["darn_normalized_importance"]
        + merged["xgb_normalized_importance"]
    ) / 2.0
    merged["mean_importance_percent"] = (
        merged["mean_normalized_importance"] * 100.0
    )
    merged["absolute_rank_difference"] = np.abs(
        merged["darn_rank"] - merged["xgb_rank"]
    )

    present_in_both = (
        (merged["darn_mean_abs_attribution"] > 0)
        & (merged["xgb_mean_abs_attribution"] > 0)
    )

    if int(present_in_both.sum()) >= 3:
        rank_x = merged.loc[present_in_both, "darn_rank"].to_numpy(float)
        rank_y = merged.loc[present_in_both, "xgb_rank"].to_numpy(float)
        spearman_rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    else:
        spearman_rho = float("nan")
    merged = merged.sort_values(
        "mean_normalized_importance",
        ascending=False,
    ).reset_index(drop=True)
    merged.insert(
        0,
        "combined_rank",
        np.arange(1, len(merged) + 1),
    )

    keep_columns = [
        "combined_rank",
        "feature",
        "pretty_feature",
        "domain",
        "darn_mean_abs_attribution",
        "xgb_mean_abs_attribution",
        "darn_importance_percent",
        "xgb_importance_percent",
        "darn_rank",
        "xgb_rank",
        "absolute_rank_difference",
        "mean_importance_percent",
    ]
    return merged[keep_columns], spearman_rho


def make_component_domain_table(
    darn_feature_table: pd.DataFrame,
    xgb_feature_table: pd.DataFrame,
) -> pd.DataFrame:
    def summarize(
        table: pd.DataFrame,
        component: str,
    ) -> pd.DataFrame:
        grouped = (
            table.groupby("domain", as_index=False)[
                "mean_absolute_direct_attribution"
            ]
            .sum()
            .rename(
                columns={
                    "mean_absolute_direct_attribution": "absolute_importance"
                }
            )
        )
        total = max(
            float(grouped["absolute_importance"].sum()),
            1e-15,
        )
        grouped["importance_percent"] = (
            grouped["absolute_importance"] / total * 100.0
        )
        grouped["component"] = component
        return grouped

    combined = pd.concat(
        [
            summarize(darn_feature_table, "Uniform DARN"),
            summarize(xgb_feature_table, "XGBoost"),
        ],
        ignore_index=True,
    )

    order = (
        combined.groupby("domain")["importance_percent"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    combined["domain_order"] = combined["domain"].map(
        {domain: index for index, domain in enumerate(order)}
    )
    return combined.sort_values(
        ["domain_order", "component"]
    ).reset_index(drop=True)


def _prepare_beeswarm_data(
    attribution_values: np.ndarray,
    feature_values: np.ndarray,
    feature_table: pd.DataFrame,
    features: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[float]]:
    top_features = (
        feature_table.head(top_k)["feature"].astype(str).tolist()
    )
    feature_to_index = {
        str(feature): index
        for index, feature in enumerate(features)
    }
    top_indices_desc = np.asarray(
        [feature_to_index[feature] for feature in top_features],
        dtype=int,
    )
    top_indices = top_indices_desc[::-1]

    attribution_top = attribution_values[:, top_indices]
    values_top = feature_values[:, top_indices]
    label_lookup = (
        feature_table.set_index("feature")["pretty_feature"]
        .astype(str)
        .to_dict()
    )
    labels = [
        label_lookup[str(features[index])]
        for index in top_indices
    ]

    feature_min = np.nanmin(values_top, axis=0)
    feature_max = np.nanmax(values_top, axis=0)
    denominator = np.where(
        feature_max - feature_min == 0,
        1.0,
        feature_max - feature_min,
    )
    values_scaled = (values_top - feature_min) / denominator

    clip_value: Optional[float] = None
    attribution_display = attribution_top

    if DISPLAY_CLIP_PERCENTILE is not None:
        clip_value = float(
            np.nanpercentile(
                np.abs(attribution_top),
                DISPLAY_CLIP_PERCENTILE,
            )
        )
        clip_value = max(clip_value, 1e-12)
        attribution_display = np.clip(
            attribution_top,
            -clip_value,
            clip_value,
        )

    return (
        attribution_display,
        values_scaled,
        labels,
        clip_value,
    )


def draw_component_beeswarm(
    ax: plt.Axes,
    attribution_values: np.ndarray,
    feature_values: np.ndarray,
    feature_table: pd.DataFrame,
    features: np.ndarray,
    *,
    top_k: int,
    title: str,
    xlabel: str,
    random_state: int,
) -> Optional[float]:
    (
        attribution_display,
        values_scaled,
        labels,
        clip_value,
    ) = _prepare_beeswarm_data(
        attribution_values,
        feature_values,
        feature_table,
        features,
        top_k,
    )

    cmap = cm.get_cmap(CMAP_NAME)
    rng = np.random.default_rng(random_state)

    ax.axvline(
        0,
        color="dimgray",
        linewidth=1.0,
        alpha=0.8,
    )

    for feature_position in range(len(labels)):
        x = attribution_display[:, feature_position]
        y = (
            np.full(len(x), feature_position, dtype=float)
            + (rng.random(len(x)) - 0.5) * BEESWARM_JITTER
        )
        colors = cmap(
            np.clip(
                values_scaled[:, feature_position],
                0.0,
                1.0,
            )
        )
        ax.scatter(
            x,
            y,
            s=7,
            c=colors,
            edgecolor="none",
            alpha=0.70,
            rasterized=True,
        )

        median_value = float(
            np.median(attribution_display[:, feature_position])
        )
        ax.plot(
            [median_value, median_value],
            [feature_position - 0.31, feature_position + 0.31],
            color="black",
            linewidth=1.5,
            alpha=0.32,
        )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel, fontsize=12.5)
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.20,
    )
    return clip_value


def plot_component_beeswarm_individual(
    attribution_values: np.ndarray,
    feature_values: np.ndarray,
    feature_table: pd.DataFrame,
    features: np.ndarray,
    output_path: pathlib.Path,
    *,
    top_k: int,
    title: str,
    xlabel: str,
    random_state: int,
) -> None:
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(1, 24)
    ax = fig.add_subplot(grid[0, :23])
    cax = fig.add_subplot(grid[0, 23])

    clip_value = draw_component_beeswarm(
        ax,
        attribution_values,
        feature_values,
        feature_table,
        features,
        top_k=top_k,
        title=title,
        xlabel=xlabel,
        random_state=random_state,
    )

    colorbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=cm.get_cmap(CMAP_NAME),
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label(
        "Transformed feature value",
        rotation=270,
        labelpad=16,
    )

    if clip_value is not None:
        ax.text(
            0.995,
            0.005,
            (
                f"Display truncated at the "
                f"{DISPLAY_CLIP_PERCENTILE:g}th percentile of "
                "|attribution|"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="dimgray",
        )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


def plot_feature_concordance(
    concordance_table: pd.DataFrame,
    spearman_rho: float,
    output_path: pathlib.Path,
    *,
    annotate_n: int = 12,
) -> None:
    epsilon = 1e-5
    x = np.maximum(
        concordance_table["darn_importance_percent"].to_numpy(float),
        epsilon,
    )
    y = np.maximum(
        concordance_table["xgb_importance_percent"].to_numpy(float),
        epsilon,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.55,
        edgecolor="none",
    )

    minimum = max(min(float(x.min()), float(y.min())), epsilon)
    maximum = max(float(x.max()), float(y.max()))
    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.0,
        color="dimgray",
    )

    for _, row in concordance_table.head(annotate_n).iterrows():
        ax.annotate(
            str(row["pretty_feature"]),
            (
                max(float(row["darn_importance_percent"]), epsilon),
                max(float(row["xgb_importance_percent"]), epsilon),
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=10.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Uniform DARN normalized importance (%)")
    ax.set_ylabel("XGBoost normalized importance (%)")
    ax.set_title(
        "C. Concordance of component-level feature importance",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.96,
        f"Spearman ρ = {spearman_rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
    )
    ax.grid(
        which="both",
        linestyle="--",
        alpha=0.20,
    )

    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


def draw_domain_comparison_axis(
    ax: plt.Axes,
    domain_table: pd.DataFrame,
    *,
    title: str,
) -> None:
    pivot = (
        domain_table.pivot(
            index="domain",
            columns="component",
            values="importance_percent",
        )
        .fillna(0.0)
    )

    order = (
        pivot.mean(axis=1)
        .sort_values(ascending=True)
        .index
        .tolist()
    )
    pivot = pivot.loc[order]

    y_positions = np.arange(len(pivot))
    height = 0.36

    ax.barh(
        y_positions - height / 2,
        pivot.get(
            "Uniform DARN",
            pd.Series(0.0, index=pivot.index),
        ),
        height=height,
        label="Uniform DARN",
        alpha=0.90,
    )
    ax.barh(
        y_positions + height / 2,
        pivot.get(
            "XGBoost",
            pd.Series(0.0, index=pivot.index),
        ),
        height=height,
        label="XGBoost",
        alpha=0.90,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Share of total mean absolute attribution (%)")
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.22,
    )
    ax.legend(loc="lower right")


def plot_domain_comparison(
    domain_table: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_domain_comparison_axis(
        ax,
        domain_table,
        title="D. Domain-level attribution by model component",
    )
    fig.tight_layout()
    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


def plot_main_component_figure(
    darn_result: Mapping[str, Any],
    darn_feature_table: pd.DataFrame,
    xgb_result: Mapping[str, Any],
    xgb_feature_table: pd.DataFrame,
    concordance_table: pd.DataFrame,
    spearman_rho: float,
    domain_table: pd.DataFrame,
    output_path: pathlib.Path,
    *,
    main_top_k: int,
) -> None:
    fig = plt.figure(figsize=(20, 15))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.25, 0.85],
        hspace=0.34,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_component_beeswarm(
        ax_a,
        darn_result["attributions"],
        darn_result["feature_values"],
        darn_feature_table,
        darn_result["features"],
        top_k=main_top_k,
        title="A. Uniform DARN: GradientSHAP",
        xlabel="GradientSHAP contribution to mortality logit",
        random_state=RANDOM_STATE,
    )

    draw_component_beeswarm(
        ax_b,
        xgb_result["attributions"],
        xgb_result["feature_values"],
        xgb_feature_table,
        xgb_result["features"],
        top_k=main_top_k,
        title="B. XGBoost: TreeSHAP",
        xlabel="TreeSHAP contribution to raw prediction margin",
        random_state=RANDOM_STATE + 11,
    )

    epsilon = 1e-5
    x = np.maximum(
        concordance_table["darn_importance_percent"].to_numpy(float),
        epsilon,
    )
    y = np.maximum(
        concordance_table["xgb_importance_percent"].to_numpy(float),
        epsilon,
    )
    ax_c.scatter(
        x,
        y,
        s=25,
        alpha=0.55,
        edgecolor="none",
    )
    minimum = max(min(float(x.min()), float(y.min())), epsilon)
    maximum = max(float(x.max()), float(y.max()))
    ax_c.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.0,
        color="dimgray",
    )
    for _, row in concordance_table.head(10).iterrows():
        ax_c.annotate(
            str(row["pretty_feature"]),
            (
                max(float(row["darn_importance_percent"]), epsilon),
                max(float(row["xgb_importance_percent"]), epsilon),
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=10.5,
        )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Uniform DARN normalized importance (%)")
    ax_c.set_ylabel("XGBoost normalized importance (%)")
    ax_c.set_title(
        "C. Feature-importance concordance",
        loc="left",
        fontweight="bold",
        fontsize=14,
    )
    ax_c.text(
        0.04,
        0.96,
        f"Spearman ρ = {spearman_rho:.2f}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
    )
    ax_c.grid(
        which="both",
        linestyle="--",
        alpha=0.20,
    )

    draw_domain_comparison_axis(
        ax_d,
        domain_table,
        title="D. Domain-level attribution",
    )

    color_scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
        cmap=cm.get_cmap(CMAP_NAME),
    )
    color_scalar.set_array([])
    colorbar = fig.colorbar(
        color_scalar,
        ax=[ax_a, ax_b],
        fraction=0.025,
        pad=0.018,
        shrink=0.92,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Transformed feature value")

    fig.suptitle(
        "Component-level explainability of the DARN–XGBoost blend",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        (
            "Attributions were calculated in an event-enriched held-out test sample "
            "and averaged across five fold-specific models. Component "
            "attributions do not constitute an exact decomposition of the "
            "final blended probability."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


# =============================================================================
# FOCUSED MAIN-MANUSCRIPT FIGURE 4: PANELS A–B ONLY
# =============================================================================

def plot_figure4_two_panel(
    darn_result: Mapping[str, Any],
    darn_feature_table: pd.DataFrame,
    xgb_result: Mapping[str, Any],
    xgb_feature_table: pd.DataFrame,
    output_path: pathlib.Path,
    *,
    top_k: int,
) -> None:
    """Generate the focused two-panel main manuscript Figure 4."""
    fig = plt.figure(figsize=(20, 10.5))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    darn_clip = draw_component_beeswarm(
        ax_a,
        darn_result["attributions"],
        darn_result["feature_values"],
        darn_feature_table,
        darn_result["features"],
        top_k=top_k,
        title="A. Uniform DARN: GradientSHAP",
        xlabel="GradientSHAP contribution to mortality logit",
        random_state=RANDOM_STATE,
    )

    xgb_clip = draw_component_beeswarm(
        ax_b,
        xgb_result["attributions"],
        xgb_result["feature_values"],
        xgb_feature_table,
        xgb_result["features"],
        top_k=top_k,
        title="B. XGBoost: TreeSHAP",
        xlabel="TreeSHAP contribution to raw prediction margin",
        random_state=RANDOM_STATE + 11,
    )

    color_scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
        cmap=cm.get_cmap(CMAP_NAME),
    )
    color_scalar.set_array([])
    colorbar = fig.colorbar(
        color_scalar,
        ax=[ax_a, ax_b],
        fraction=0.022,
        pad=0.018,
        shrink=0.90,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label(
        "Transformed feature value",
        rotation=270,
        labelpad=20,
        fontsize=12.5,
    )
    colorbar.ax.tick_params(labelsize=11.5)

    fig.suptitle(
        "Component-level explainability of Uniform DARN and XGBoost",
        fontsize=18,
        fontweight="bold",
        y=0.982,
    )

    clipping_note = ""
    if darn_clip is not None or xgb_clip is not None:
        clipping_note = (
            f" Display truncated at the {DISPLAY_CLIP_PERCENTILE:g}th "
            "percentile of absolute attribution within each panel."
        )


    fig.subplots_adjust(
        left=0.15,
        right=0.93,
        top=0.935,
        bottom=0.10,
    )

    save_figure_all_formats(fig, output_path)
    plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def build_explainability(
    run_dir: pathlib.Path,
    pipeline_module_path: pathlib.Path,
    data_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    top_k: int,
    main_top_k: int,
    patient_top_k: int,
    max_samples: int,
    background_n: int,
    gradient_samples: int,
    attribution_batch_size: int,
    force_recompute: bool,
    device_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    print("=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 4")
    print("=" * 100)
    print("Run directory   :", run_dir)
    print("Pipeline module :", pipeline_module_path)
    print("Output directory:", output_dir)

    module = load_pipeline_module(pipeline_module_path)
    reconstructed = reconstruct_current_run(
        module,
        run_dir,
        data_path,
    )

    y_test = reconstructed["y_test"]
    test_predictions = reconstructed["test_predictions"]

    darn_probability_column = "uniform_darn_calibrated_probability"
    if darn_probability_column not in test_predictions.columns:
        candidates = [
            column
            for column in test_predictions.columns
            if "uniform_darn" in column.lower()
            and column.endswith("_calibrated_probability")
        ]
        if not candidates:
            raise KeyError(
                "Could not locate the calibrated Uniform DARN probability "
                "column in test_predictions_all_models.csv."
            )
        darn_probability_column = candidates[0]

    uniform_darn_probability = test_predictions[
        darn_probability_column
    ].astype(float).to_numpy()

    requested_sample_indices = select_attribution_sample(
        y_test,
        max_samples=max_samples,
        random_state=RANDOM_STATE,
    )

    if device_name == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)

    # Attribution caches are intentionally scoped to this exact strict-OOF
    # run and this output directory. Never reuse a cache from an older fitted
    # model, because fold-specific feature selection and neural parameters may
    # differ even when the cohort definition is similar.
    darn_cache_path = (
        output_dir / "direct_uniform_darn_ensemble_attributions.npz"
    )

    darn_result = compute_current_run_attributions(
        module,
        reconstructed,
        run_dir,
        requested_sample_indices,
        background_n=background_n,
        gradient_samples=gradient_samples,
        attribution_batch_size=attribution_batch_size,
        device=device,
        cache_path=darn_cache_path,
        force_recompute=force_recompute,
    )

    sample_indices = darn_result["sample_indices"].astype(int)
    if not np.array_equal(
        sample_indices,
        requested_sample_indices,
    ):
        print(
            "Using the sample stored in the Uniform DARN attribution cache."
        )

    xgb_cache_path = (
        output_dir / "xgboost_treeshap_ensemble_attributions.npz"
    )
    xgb_result = compute_current_run_xgboost_attributions(
        reconstructed,
        run_dir,
        sample_indices,
        cache_path=xgb_cache_path,
        force_recompute=force_recompute,
    )

    if not np.array_equal(
        xgb_result["sample_indices"],
        sample_indices,
    ):
        raise RuntimeError(
            "Uniform DARN and XGBoost explanations use different samples."
        )

    sample_columns = [
        "final_cohort_row_position",
        "source_row_position",
        "y_true",
        darn_probability_column,
    ]
    xgb_probability_candidates = [
        column
        for column in test_predictions.columns
        if "xgboost" in column.lower()
        and column.endswith("_calibrated_probability")
    ]
    if xgb_probability_candidates:
        sample_columns.append(xgb_probability_candidates[0])

    sample_table = test_predictions.iloc[sample_indices][
        list(dict.fromkeys(sample_columns))
    ].copy()
    sample_table.insert(
        0,
        "test_row_position",
        sample_indices,
    )
    sample_table.to_csv(
        output_dir / "component_explainability_sample.csv",
        index=False,
    )

    n_folds = int(reconstructed["config"].get("n_folds", 5))

    darn_feature_table, darn_domain_raw = make_importance_tables(
        darn_result["features"],
        darn_result["domains"],
        darn_result["attributions"],
        darn_result["presence_count"],
        n_folds=n_folds,
    )
    xgb_feature_table, xgb_domain_raw = make_importance_tables(
        xgb_result["features"],
        xgb_result["domains"],
        xgb_result["attributions"],
        xgb_result["presence_count"],
        n_folds=n_folds,
    )

    darn_display_table = feature_table_for_display(
        darn_feature_table
    )
    xgb_display_table = feature_table_for_display(
        xgb_feature_table
    )

    darn_filtered_missing = darn_feature_table.loc[
        darn_feature_table["is_missing_indicator"].astype(bool)
    ].copy()
    darn_filtered_missing.insert(
        0,
        "component",
        "Uniform DARN",
    )

    xgb_filtered_missing = xgb_feature_table.loc[
        xgb_feature_table["is_missing_indicator"].astype(bool)
    ].copy()
    xgb_filtered_missing.insert(
        0,
        "component",
        "XGBoost",
    )

    filtered_missing_table = pd.concat(
        [
            darn_filtered_missing,
            xgb_filtered_missing,
        ],
        ignore_index=True,
    )
    filtered_missing_table.to_csv(
        output_dir / "Figure4_missing_indicators_excluded_from_display.csv",
        index=False,
    )

    if len(darn_display_table) < main_top_k:
        raise RuntimeError(
            "Too few non-missing-indicator Uniform DARN features remain "
            f"for main_top_k={main_top_k}."
        )
    if len(xgb_display_table) < main_top_k:
        raise RuntimeError(
            "Too few non-missing-indicator XGBoost features remain "
            f"for main_top_k={main_top_k}."
        )

    _, darn_label_audit = make_unique_feature_labels(
        darn_result["features"].astype(str)
    )
    _, xgb_label_audit = make_unique_feature_labels(
        xgb_result["features"].astype(str)
    )

    darn_label_audit.insert(
        0,
        "component",
        "Uniform DARN",
    )
    xgb_label_audit.insert(
        0,
        "component",
        "XGBoost",
    )

    label_audit = pd.concat(
        [
            darn_label_audit,
            xgb_label_audit,
        ],
        ignore_index=True,
    )
    label_audit.to_csv(
        output_dir / "Figure4_feature_label_audit.csv",
        index=False,
    )

    assert_unique_display_labels(
        darn_display_table,
        top_k=main_top_k,
        component_name="Uniform DARN Panel A",
    )
    assert_unique_display_labels(
        xgb_display_table,
        top_k=main_top_k,
        component_name="XGBoost Panel B",
    )

    darn_feature_table.to_csv(
        output_dir / "uniform_darn_gradientshap_feature_importance.csv",
        index=False,
    )
    xgb_feature_table.to_csv(
        output_dir / "xgboost_treeshap_feature_importance.csv",
        index=False,
    )
    darn_domain_raw.to_csv(
        output_dir / "uniform_darn_gradientshap_domain_importance_raw.csv",
        index=False,
    )
    xgb_domain_raw.to_csv(
        output_dir / "xgboost_treeshap_domain_importance_raw.csv",
        index=False,
    )

    concordance_table, spearman_rho = (
        make_component_concordance_table(
            darn_feature_table,
            xgb_feature_table,
        )
    )
    concordance_table.to_csv(
        output_dir / "darn_xgboost_feature_concordance.csv",
        index=False,
    )

    component_domain_table = make_component_domain_table(
        darn_feature_table,
        xgb_feature_table,
    )
    component_domain_table.to_csv(
        output_dir / "darn_xgboost_domain_importance_percent.csv",        index=False,
    )

    print("\n" + "=" * 100)
    print("TOP DIRECT UNIFORM-DARN FEATURES")
    print("=" * 100)
    print(
        darn_display_table[
            [
                "rank",
                "pretty_feature",
                "domain",
                "mean_absolute_direct_attribution",
                "fold_presence_count",
            ]
        ]
        .head(top_k)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("TOP XGBOOST TREESHAP FEATURES")
    print("=" * 100)
    print(
        xgb_display_table[
            [
                "rank",
                "pretty_feature",
                "domain",
                "mean_absolute_direct_attribution",
                "fold_presence_count",
            ]
        ]
        .head(top_k)
        .to_string(index=False)
    )

    print("\nComponent feature-rank concordance:")
    print(f"Spearman rho = {spearman_rho:.4f}")

    print("\n" + "=" * 100)
    print("MISSINGNESS INDICATORS EXCLUDED FROM MANUSCRIPT DISPLAY")
    print("=" * 100)
    print(
        f"Uniform DARN excluded: {len(darn_filtered_missing):,}"
    )
    print(
        f"XGBoost excluded:      {len(xgb_filtered_missing):,}"
    )
    if not filtered_missing_table.empty:
        print(
            filtered_missing_table[
                [
                    "component",
                    "feature",
                    "pretty_feature",
                    "domain",
                    "mean_absolute_direct_attribution",
                ]
            ]
            .sort_values(
                [
                    "component",
                    "mean_absolute_direct_attribution",
                ],
                ascending=[True, False],
            )
            .to_string(index=False)
        )

    duplicated_label_rows = label_audit.loc[
        label_audit["base_label_was_duplicated"]
    ]
    if not duplicated_label_rows.empty:
        print("\n" + "=" * 100)
        print("DISAMBIGUATED FIGURE LABELS")
        print("=" * 100)
        print(
            duplicated_label_rows[
                [
                    "component",
                    "feature",
                    "base_pretty_label",
                    "final_pretty_label",
                ]
            ].to_string(index=False)
        )

    plot_component_beeswarm_individual(
        darn_result["attributions"],
        darn_result["feature_values"],
        darn_display_table,
        darn_result["features"],
        output_dir / "Figure4A_Uniform_DARN_Direct_GradientSHAP.png",
        top_k=main_top_k,
        title="A. Uniform DARN: GradientSHAP",
        xlabel="GradientSHAP contribution to mortality logit",
        random_state=RANDOM_STATE,
    )

    plot_component_beeswarm_individual(
        xgb_result["attributions"],
        xgb_result["feature_values"],
        xgb_display_table,
        xgb_result["features"],
        output_dir / "Figure4B_XGBoost_TreeSHAP.png",
        top_k=main_top_k,
        title="B. XGBoost: TreeSHAP",
        xlabel="TreeSHAP contribution to raw prediction margin",
        random_state=RANDOM_STATE + 11,
    )

    main_figure_path = (
        output_dir
        / "Figure4_DARN_XGBoost_Component_Explainability_AB.png"
    )

    plot_figure4_two_panel(
        darn_result,
        darn_display_table,
        xgb_result,
        xgb_display_table,
        main_figure_path,
        top_k=main_top_k,
    )

    # Feature-rank concordance, domain contribution and patient-level
    # explanations are intentionally excluded from the main Figure 4.
    # They can be generated separately for supplementary or domain figures.
    patient_metadata = {}

    summary = {
        "analysis_version": "temporal-safe-strict-oof",
        "run_id": run_dir.name,
        "run_directory": str(run_dir),
        "pipeline_module": str(pipeline_module_path),
        "data_path": str(reconstructed["data_path"]),
        "cohort_exclusion": (
            "ASA physical status 6 organ-donor encounters excluded before "
            "model splitting"
        ),
        "interpretation": (
            "Focused component-level explanation of the saved Uniform DARN "
            "and XGBoost models. These component attributions are not an exact "
            "decomposition of either hybrid prediction."
        ),
        "main_figure_panels": {
            "A": "Direct GradientSHAP for Uniform DARN",
            "B": "Exact TreeSHAP for XGBoost",
        },
        "uniform_darn_method": (
            "Direct GradientSHAP averaged across five fold-specific "
            "Uniform DARN models"
        ),
        "uniform_darn_scale": "Raw mortality logit",
        "xgboost_method": (
            "Exact TreeSHAP from XGBoost pred_contribs, averaged across "
            "five fold-specific XGBoost models"
        ),
        "xgboost_scale": "Raw XGBoost prediction margin",
        "important_limitation": (
            "Component-level attributions do not provide an exact additive "
            "decomposition of the final calibrated blended probability."
        ),
        "test_n": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "attribution_sample_n": int(len(sample_indices)),
        "attribution_sample_events": int(y_test[sample_indices].sum()),
        "event_enriched_sample": True,
        "n_folds": n_folds,
        "uniform_darn_union_features": int(
            len(darn_result["features"])
        ),
        "xgboost_union_features": int(
            len(xgb_result["features"])
        ),
        "feature_rank_spearman_rho": float(spearman_rho),
        "display_clip_percentile": float(
            DISPLAY_CLIP_PERCENTILE
        ),
        "top_k_individual": int(top_k),
        "top_k_main_figure": int(main_top_k),
        "feature_label_audit": str(
            output_dir / "Figure4_feature_label_audit.csv"
        ),
        "exclude_missing_indicators_from_display": bool(
            EXCLUDE_MISSING_INDICATORS_FROM_DISPLAY
        ),
        "missing_indicator_display_audit": str(
            output_dir / "Figure4_missing_indicators_excluded_from_display.csv"
        ),
        "uniform_darn_missing_indicators_excluded_from_display": int(
            len(darn_filtered_missing)
        ),
        "xgboost_missing_indicators_excluded_from_display": int(
            len(xgb_filtered_missing)
        ),
        "complete_feature_importance_tables_retain_missing_indicators": True,
        "display_labels_unique_within_each_main_panel": True,
        "duplicated_base_labels_disambiguated": int(
            label_audit["base_label_was_duplicated"].sum()
        ),
        "background_n_per_darn_fold": int(background_n),
        "gradient_samples": int(gradient_samples),
        "device": str(device),
        "main_figure": str(main_figure_path),
    }

    save_json(
        summary,
        output_dir / "component_explainability_summary.json",
    )

    print("\n" + "=" * 100)
    print("FINAL TEMPORAL-SAFE STRICT-OOF DARN-XGBOOST EXPLAINABILITY COMPLETE")
    print("=" * 100)
    print("Output directory:", output_dir)
    print("\nMain figure:")
    print(main_figure_path)
    print("\nInterpretation note:")
    print(
        "Use the DARN and XGBoost results as component-level explanations "
        "of the primary hybrid. Do not describe either component alone, or "
        "their normalized importance comparison, as an exact SHAP "
        "decomposition of the final blended probability."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Focused two-panel Figure 4 for the temporal-safe strict-OOF DARN and XGBoost "
            "model using direct GradientSHAP and exact TreeSHAP from the "
            "saved fold-specific models."
        )
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
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--main_top_k", type=int, default=MAIN_TOP_K)
    parser.add_argument(
        "--patient_top_k",
        type=int,
        default=PATIENT_TOP_K,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_EXPLAIN_N,
    )
    parser.add_argument(
        "--background_n",
        type=int,
        default=BACKGROUND_N,
    )
    parser.add_argument(
        "--gradient_samples",
        type=int,
        default=GRADIENT_SAMPLES,
    )
    parser.add_argument(
        "--attribution_batch_size",
        type=int,
        default=ATTRIBUTION_BATCH_SIZE,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or a device such as cuda:0",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help=(
            "Ignore saved strict-OOF attribution caches and recompute both "
            "Uniform DARN GradientSHAP and XGBoost TreeSHAP."
        ),
    )

    # Jupyter/IPython injects a kernel connection argument such as
    # `--f=/path/kernel.json` or `-f /path/kernel.json`. With argparse's
    # default abbreviation behavior, `--f` may be mistaken for
    # `--force_recompute`, so abbreviations are disabled above and these
    # notebook-only arguments are removed explicitly.
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    filtered_argv: List[str] = []
    skip_next = False

    for argument in raw_argv:
        if skip_next:
            skip_next = False
            continue

        if argument in {"-f", "--f"}:
            skip_next = True
            continue

        if argument.startswith("-f=") or argument.startswith("--f="):
            continue

        filtered_argv.append(argument)

    args, unknown = parser.parse_known_args(filtered_argv)

    if unknown:
        print(
            "Ignoring unrecognized Jupyter/IPython arguments:",
            " ".join(map(str, unknown)),
        )

    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.run_dir
        / "figures_revised_manuscript"
        / "figure4_darn_xgboost_AB_strict_oof"
    )

    build_explainability(
        run_dir=args.run_dir,
        pipeline_module_path=args.pipeline_module,
        data_path=args.data_path,
        output_dir=output_dir,
        top_k=args.top_k,
        main_top_k=args.main_top_k,
        patient_top_k=args.patient_top_k,
        max_samples=args.max_samples,
        background_n=args.background_n,
        gradient_samples=args.gradient_samples,
        attribution_batch_size=args.attribution_batch_size,
        force_recompute=args.force_recompute,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()