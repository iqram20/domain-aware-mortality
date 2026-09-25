#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

FIGURE 6 — SUBGROUP PERFORMANCE AND CALIBRATION

==========================================================



Corrected run:

/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/

darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653



Current main-figure plan

------------------------

A. AUROC across prespecified clinical subgroups.

B. AUPRC across subgroups, compared with each subgroup's own prevalence.

C. Calibration-in-the-large using observed/expected mortality ratio.

D. Sensitivity and false-positive rate at the development-defined threshold.



Reference model for this subgroup audit: Hybrid DARN-XGB Blend.



The older elective/emergency PR panel is saved separately as supplementary.

The older three-seed panel is removed because the strict-OOF analysis used

five fold-specific models rather than a three-seed repeated-training design.



This is an exploratory internal subgroup audit. The held-out test cohort contains

47 deaths, so small-event strata and wide intervals must be interpreted with

caution and should not be described as proof of fairness.

"""



from __future__ import annotations



import argparse

import importlib.util

import json

import pathlib

import re

import sys

import warnings

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

from matplotlib import cm

from matplotlib.lines import Line2D

from sklearn.metrics import (

    average_precision_score,

    brier_score_loss,

    precision_recall_curve,

    roc_auc_score,

)



warnings.filterwarnings("ignore")



# =============================================================================

# CURRENT RUN SETTINGS

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

    "figures_revised_manuscript/figure6_subgroup_performance_strict_oof"

)



PRIMARY_MODEL = "Hybrid DARN-XGB Blend"

THRESHOLD_LABEL = "development-defined operating point"

FIGURE_BASENAME = "Figure6_TemporalSafe_StrictOOF_Subgroup_Performance_2x2"

SUPPLEMENTARY_BASENAME = "Supplementary_Figure6_Urgency_PR_Curves"



CMAP_NAME = "plasma"

N_BOOTSTRAP = 2000

N_JOBS = 8

RANDOM_STATE = 20260919

MIN_N_ANALYSIS = 30

MIN_EVENTS_DISPLAY = 5



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

VARIABLE_ORDER = ["Overall", "Sex", "Age", "ASA", "Urgency"]



# =============================================================================

# HELPERS

# =============================================================================



def require_file(path: pathlib.Path) -> pathlib.Path:

    if not path.exists():

        raise FileNotFoundError(

            f"Required current-run file not found:\n{path}\n"

            "Confirm the completed strict-OOF run directory and paths."

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

        raise TypeError(type(value))



    with open(path, "w", encoding="utf-8") as handle:

        json.dump(data, handle, indent=2, default=convert)





def first_existing_column(

    frame: pd.DataFrame,

    candidates: Sequence[str],

) -> Optional[str]:

    lower = {str(c).lower(): str(c) for c in frame.columns}

    for candidate in candidates:

        if candidate in frame.columns:

            return candidate

        match = lower.get(candidate.lower())

        if match is not None:

            return match

    return None





def safe_model_name(model_name: str) -> str:

    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()





def load_pipeline_module(path: pathlib.Path):

    require_file(path)

    spec = importlib.util.spec_from_file_location(

        "mortality_darn_v6_figure6_module", path

    )

    if spec is None or spec.loader is None:

        raise ImportError(f"Could not import pipeline module: {path}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module





def save_figure_all_formats(fig: plt.Figure, png_path: pathlib.Path) -> None:

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



# =============================================================================

# HOLDOUT RECONSTRUCTION

# =============================================================================



def reconstruct_holdout(

    module,

    run_dir: pathlib.Path,

    data_path: pathlib.Path,

) -> Dict[str, Any]:

    config = load_json(run_dir / "config.json")

    configured = pathlib.Path(str(config.get("data_path", data_path)))

    if data_path == DEFAULT_DATA_PATH and configured.exists():

        data_path = configured



    prediction_file = require_file(run_dir / "test_predictions_all_models.csv")

    predictions = pd.read_csv(prediction_file)

    if "y_true" not in predictions.columns:

        raise KeyError("test_predictions_all_models.csv is missing y_true.")



    source = pd.read_csv(require_file(data_path), low_memory=False)

    target = str(config.get("target_column", "mortality_30d"))

    cohort, flow = module.apply_reviewer_cohort_exclusions(

        source,

        target_column=target,

        exclude_asa6=bool(config.get("exclude_asa6", True)),

    )



    position_col = first_existing_column(

        predictions,

        ["final_cohort_row_position", "original_row_position"],

    )

    if position_col is not None:

        positions = pd.to_numeric(

            predictions[position_col], errors="raise"

        ).astype(int).to_numpy()

        if np.any(positions < 0) or np.any(positions >= len(cohort)):

            raise IndexError(f"Invalid positions in {position_col}.")

        holdout = cohort.iloc[positions].reset_index(drop=True)

    elif "source_row_position" in predictions.columns:

        positions = pd.to_numeric(

            predictions["source_row_position"], errors="raise"

        ).astype(int).to_numpy()

        source_indexed = source.reset_index(drop=False).rename(

            columns={"index": "source_row_position"}

        )

        holdout = (

            source_indexed.set_index("source_row_position")

            .loc[positions]

            .reset_index(drop=True)

        )

        position_col = "source_row_position"

    else:

        raise KeyError(

            "Prediction file must contain final_cohort_row_position, "

            "original_row_position, or source_row_position."

        )



    y_saved = predictions["y_true"].astype(int).to_numpy()

    y_reconstructed = pd.to_numeric(

        holdout[target], errors="raise"

    ).astype(int).to_numpy()

    if not np.array_equal(y_saved, y_reconstructed):

        raise RuntimeError(

            "Reconstructed holdout outcomes do not match saved predictions."

        )



    print("\n" + "=" * 100)

    print("STRICT-OOF TEST-SET ALIGNMENT")

    print("=" * 100)

    print(flow.to_string(index=False))

    print(f"Test N:      {len(y_saved):,}")

    print(f"Test deaths: {int(y_saved.sum()):,}")

    print(f"Position field: {position_col}")



    return {

        "config": config,

        "data_path": data_path,

        "predictions": predictions,

        "holdout": holdout,

        "y_true": y_saved,

    }





def load_primary_model(

    run_dir: pathlib.Path,

    predictions: pd.DataFrame,

    model_name: str,

) -> Dict[str, Any]:

    metrics = pd.read_csv(require_file(run_dir / "test_model_metrics.csv"))

    if "model" not in metrics.columns:

        raise KeyError("test_model_metrics.csv is missing model.")



    selected = metrics.loc[metrics["model"].astype(str).eq(model_name)]

    if selected.empty:

        raise KeyError(

            f"Audit model '{model_name}' not found. Available models: "

            f"{metrics['model'].astype(str).tolist()}"

        )

    row = selected.iloc[0]

    threshold_col = first_existing_column(metrics, ["Threshold", "threshold"])

    if threshold_col is None:

        raise KeyError("test_model_metrics.csv is missing Threshold.")

    threshold = float(row[threshold_col])



    safe = safe_model_name(model_name)

    probability_col = first_existing_column(

        predictions,

        [

            f"{safe}_calibrated_probability",

            f"{safe}_probability",

            f"prob_{safe}",

        ],

    )

    if probability_col is None:

        raise KeyError(

            f"Could not locate the calibrated probability column for "

            f"'{model_name}'. Available columns: {list(predictions.columns)}"

        )



    probability = pd.to_numeric(

        predictions[probability_col], errors="raise"

    ).astype(float).to_numpy()

    if not np.isfinite(probability).all():

        raise ValueError("Audit-model probabilities contain non-finite values.")



    return {

        "metrics": metrics,

        "model_row": row,

        "model_name": model_name,

        "probability_column": probability_col,

        "probability": probability,

        "threshold": threshold,

    }



# =============================================================================

# SUBGROUPS

# =============================================================================



def normalize_sex(series: pd.Series) -> pd.Series:

    raw = series.astype(str).str.strip().str.lower()

    output = pd.Series("Missing/Other", index=series.index, dtype=object)

    output.loc[raw.isin({"f", "female", "woman", "women", "2", "0"})] = "Female"

    output.loc[raw.isin({"m", "male", "man", "men", "1"})] = "Male"

    output.loc[raw.str.contains("female", na=False)] = "Female"

    output.loc[

        raw.str.contains("male", na=False)

        & ~raw.str.contains("female", na=False)

    ] = "Male"

    return output





def normalize_emergency(series: pd.Series) -> pd.Series:

    raw = series.astype(str).str.strip().str.lower()

    output = pd.Series("Missing/Other", index=series.index, dtype=object)

    output.loc[

        raw.isin({"1", "true", "yes", "y", "emergency", "emergent", "urgent", "e"})

    ] = "Emergency"

    output.loc[

        raw.isin({"0", "false", "no", "n", "elective", "scheduled", "non-emergency", "nonemergency"})

    ] = "Elective"

    numeric = pd.to_numeric(series, errors="coerce")

    output.loc[numeric.eq(1)] = "Emergency"

    output.loc[numeric.eq(0)] = "Elective"

    output.loc[

        raw.str.contains("emerg", na=False)

        | raw.str.contains("urgent", na=False)

    ] = "Emergency"

    output.loc[

        raw.str.contains("elect", na=False)

        | raw.str.contains("sched", na=False)

    ] = "Elective"

    return output





def parse_asa_numeric(series: pd.Series) -> pd.Series:

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() > 0:

        return numeric.astype(float)

    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}



    def parse_one(value: Any) -> float:

        if pd.isna(value):

            return np.nan

        text = str(value).strip().lower()

        match = re.search(r"\b(vi|iv|v|iii|ii|i)\b", text)

        if match:

            return float(roman[match.group(1)])

        match = re.search(r"([1-6])", text)

        return float(match.group(1)) if match else np.nan



    return series.map(parse_one).astype(float)





def derive_subgroups(holdout: pd.DataFrame) -> Dict[str, pd.Series]:

    groups: Dict[str, pd.Series] = {}

    sex_col = first_existing_column(holdout, ["sex", "gender"])

    age_col = first_existing_column(holdout, ["age"])

    asa_col = first_existing_column(

        holdout, ["asa", "asa_class", "asa_status", "asa_ps"]

    )

    urgency_col = first_existing_column(

        holdout, ["emop", "emergency", "emergency_status", "urgent"]

    )



    if sex_col is not None:

        groups["Sex"] = normalize_sex(holdout[sex_col])



    if age_col is not None:

        age = pd.to_numeric(holdout[age_col], errors="coerce")

        result = pd.Series("Missing/Other", index=holdout.index, dtype=object)

        result.loc[age.lt(60)] = "<60"

        result.loc[age.ge(60) & age.lt(70)] = "60–69"

        result.loc[age.ge(70)] = "≥70"

        groups["Age"] = result



    if asa_col is not None:

        asa = parse_asa_numeric(holdout[asa_col])

        if asa.eq(6).any():

            raise RuntimeError("ASA 6 remains in the corrected holdout.")

        result = pd.Series("Missing/Other", index=holdout.index, dtype=object)

        result.loc[asa.isin([1, 2])] = "I–II"

        result.loc[asa.eq(3)] = "III"

        result.loc[asa.isin([4, 5])] = "IV–V"

        groups["ASA"] = result



    if urgency_col is not None:

        groups["Urgency"] = normalize_emergency(holdout[urgency_col])



    missing = {"Sex", "Age", "ASA", "Urgency"} - set(groups)

    if missing:

        print("Warning: subgroup variables unavailable:", sorted(missing))

    return groups



# =============================================================================

# METRICS AND BOOTSTRAP

# =============================================================================



def safe_divide(numerator: float, denominator: float) -> float:

    return np.nan if denominator == 0 else float(numerator / denominator)





def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:

    y = np.asarray(y, dtype=int)

    p = np.asarray(p, dtype=float)

    predicted = (p >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (y == 1)))

    fp = int(np.sum((predicted == 1) & (y == 0)))

    tn = int(np.sum((predicted == 0) & (y == 0)))

    fn = int(np.sum((predicted == 0) & (y == 1)))

    prevalence = float(y.mean())

    observed = float(y.sum())

    expected = float(p.sum())



    result: Dict[str, float] = {

        "n": int(len(y)),

        "events": int(observed),

        "nonevents": int(len(y) - observed),

        "prevalence": prevalence,

        "AUPRC_baseline": prevalence,

        "mean_predicted_risk": float(p.mean()),

        "Brier": float(brier_score_loss(y, p)),

        "TP": tp,

        "FP": fp,

        "TN": tn,

        "FN": fn,

        "sensitivity": safe_divide(tp, tp + fn),

        "specificity": safe_divide(tn, tn + fp),

        "FPR": safe_divide(fp, fp + tn),

        "precision_PPV": safe_divide(tp, tp + fp),

        "observed_expected_ratio": safe_divide(observed, expected),

        "alerts_per_1000": float(1000.0 * (tp + fp) / len(y)),

    }

    if len(np.unique(y)) == 2:

        result["AUROC"] = float(roc_auc_score(y, p))

        result["AUPRC"] = float(average_precision_score(y, p))

        result["AUPRC_fold_over_baseline"] = safe_divide(

            result["AUPRC"], prevalence

        )

    else:

        result["AUROC"] = np.nan

        result["AUPRC"] = np.nan

        result["AUPRC_fold_over_baseline"] = np.nan

    return result





def percentile_interval(values: Sequence[float]) -> Tuple[float, float]:

    array = np.asarray(values, dtype=float)

    array = array[np.isfinite(array)]

    if len(array) == 0:

        return np.nan, np.nan

    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))





def bootstrap_metrics(

    y: np.ndarray,

    p: np.ndarray,

    threshold: float,

    n_bootstrap: int,

    seed: int,

    n_jobs: int,

) -> Dict[str, Tuple[float, float, int]]:

    y = np.asarray(y, dtype=int)

    p = np.asarray(p, dtype=float)

    n = len(y)

    metric_names = [

        "AUROC",

        "AUPRC",

        "AUPRC_fold_over_baseline",

        "sensitivity",

        "FPR",

        "precision_PPV",

        "observed_expected_ratio",

        "Brier",

    ]

    sequence = np.random.SeedSequence(seed)

    seeds = [int(child.generate_state(1)[0]) for child in sequence.spawn(n_bootstrap)]



    def one(rep_seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(rep_seed)

        indices = rng.integers(0, n, size=n)

        return point_metrics(y[indices], p[indices], threshold)



    results = Parallel(n_jobs=n_jobs, prefer="threads")(

        delayed(one)(rep_seed) for rep_seed in seeds

    )

    output: Dict[str, Tuple[float, float, int]] = {}

    for metric in metric_names:

        values = np.asarray([row.get(metric, np.nan) for row in results], dtype=float)

        low, high = percentile_interval(values)

        output[metric] = (low, high, int(np.isfinite(values).sum()))

    return output





def calculate_subgroup_table(

    y: np.ndarray,

    p: np.ndarray,

    threshold: float,

    groups: Mapping[str, pd.Series],

    n_bootstrap: int,

    n_jobs: int,

) -> pd.DataFrame:

    definitions: List[Dict[str, Any]] = [

        {

            "subgroup_variable": "Overall",

            "subgroup_level": "Overall",

            "mask": np.ones(len(y), dtype=bool),

        }

    ]

    for variable, series in groups.items():

        series = series.reset_index(drop=True)

        for level in pd.unique(series):

            if str(level) == "Missing/Other":

                continue

            definitions.append(

                {

                    "subgroup_variable": variable,

                    "subgroup_level": str(level),

                    "mask": series.eq(level).to_numpy(),

                }

            )



    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 100)

    print("FINAL STRICT-OOF SUBGROUP BOOTSTRAP")

    print("=" * 100)



    for index, definition in enumerate(definitions):

        mask = np.asarray(definition["mask"], dtype=bool)

        yg, pg = y[mask], p[mask]

        if len(yg) < MIN_N_ANALYSIS:

            continue

        print(

            f"[{index + 1:02d}/{len(definitions):02d}] "

            f"{definition['subgroup_variable']}: {definition['subgroup_level']} | "

            f"N={len(yg):,}, deaths={int(yg.sum())}"

        )

        point = point_metrics(yg, pg, threshold)

        intervals = bootstrap_metrics(

            yg,

            pg,

            threshold,

            n_bootstrap=n_bootstrap,

            seed=RANDOM_STATE + 1009 * (index + 1),

            n_jobs=n_jobs,

        )

        row: Dict[str, Any] = {

            "subgroup_variable": definition["subgroup_variable"],

            "subgroup_level": definition["subgroup_level"],

            **point,

        }

        for metric, (low, high, valid) in intervals.items():

            row[f"{metric}_ci_lower_95"] = low

            row[f"{metric}_ci_upper_95"] = high

            row[f"{metric}_valid_bootstraps"] = valid

        row["low_event_flag"] = int(point["events"] < MIN_EVENTS_DISPLAY)

        rows.append(row)



    table = pd.DataFrame(rows)

    if table.empty:

        raise RuntimeError("No subgroup results were generated.")



    order_map = {pair: idx for idx, pair in enumerate(DISPLAY_ORDER)}

    table["display_order"] = [

        order_map.get((row.subgroup_variable, row.subgroup_level), 999)

        for row in table.itertuples()

    ]

    table["display_label"] = table.apply(

        lambda row: (

            "Overall"

            if row["subgroup_variable"] == "Overall"

            else f"{row['subgroup_variable']}: {row['subgroup_level']}"

        ),

        axis=1,

    )

    table["display_label_with_counts"] = table["display_label"] + table.apply(

        lambda row: f"  (N={int(row['n']):,}; deaths={int(row['events'])})",

        axis=1,

    )

    return table





def calculate_gap_table(table: pd.DataFrame) -> pd.DataFrame:

    metrics = [

        "AUROC",

        "AUPRC",

        "AUPRC_fold_over_baseline",

        "sensitivity",

        "FPR",

        "precision_PPV",

        "observed_expected_ratio",

        "Brier",

    ]

    rows: List[Dict[str, Any]] = []

    for variable, subset in table.groupby("subgroup_variable"):

        if variable == "Overall":

            continue

        for metric in metrics:

            values = pd.to_numeric(subset[metric], errors="coerce").dropna()

            if len(values) < 2:

                continue

            min_idx, max_idx = values.idxmin(), values.idxmax()

            rows.append(

                {

                    "subgroup_variable": variable,

                    "metric": metric,

                    "minimum": float(values.min()),

                    "minimum_group": str(table.loc[min_idx, "subgroup_level"]),

                    "maximum": float(values.max()),

                    "maximum_group": str(table.loc[max_idx, "subgroup_level"]),

                    "absolute_gap": float(values.max() - values.min()),

                }

            )

    return pd.DataFrame(rows)



# =============================================================================

# PLOTTING

# =============================================================================



def configure_style() -> None:
    """Use the same publication typography as Figures 2–5."""
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
            "figure.titlesize": 18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )



def variable_colors() -> Dict[str, Any]:

    cmap = cm.get_cmap(CMAP_NAME)

    positions = np.linspace(0.10, 0.90, len(VARIABLE_ORDER))

    return {name: cmap(pos) for name, pos in zip(VARIABLE_ORDER, positions)}





def forest_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    metric: str,

    title: str,

    xlabel: str,

    colors: Mapping[str, Any],

    xlim: Optional[Tuple[float, float]] = None,

    reference: Optional[float] = None,

    show_ylabels: bool = True,

    prevalence_markers: bool = False,

) -> None:

    plot_table = table.iloc[::-1].reset_index(drop=True)

    for idx, row in plot_table.iterrows():

        estimate = float(row[metric])

        low = float(row[f"{metric}_ci_lower_95"])

        high = float(row[f"{metric}_ci_upper_95"])

        if not np.isfinite(estimate):

            continue

        low_event = int(row["events"]) < MIN_EVENTS_DISPLAY

        overall = row["subgroup_variable"] == "Overall"

        xerr = None

        if np.isfinite(low) and np.isfinite(high):

            xerr = np.asarray([[estimate - low], [high - estimate]])

        ax.errorbar(

            estimate,

            idx,

            xerr=xerr,

            fmt="X" if low_event else "o",

            markersize=7.5 if overall else 6.0,

            capsize=2.4,

            linewidth=2.0 if overall else 1.25,

            color=colors.get(row["subgroup_variable"], cm.get_cmap(CMAP_NAME)(0.5)),

            alpha=0.45 if low_event else 1.0,

            markeredgecolor="white",

            markeredgewidth=0.45,

            zorder=3,

        )

        if prevalence_markers:

            ax.scatter(

                float(row["prevalence"]),

                idx,

                marker="|",

                s=80,

                linewidths=1.8,

                color="dimgray",

                alpha=0.85,

                zorder=2,

            )



    if reference is not None:

        ax.axvline(reference, linestyle=":", linewidth=1.0, color="gray")

    ax.set_yticks(np.arange(len(plot_table)))

    if show_ylabels:

        ax.set_yticklabels(plot_table["display_label_with_counts"], fontsize=11.5)

    else:

        ax.set_yticklabels([])

        ax.tick_params(axis="y", left=False)

    if xlim is not None:

        ax.set_xlim(*xlim)

    ax.set_ylim(-0.6, len(plot_table) - 0.4)

    ax.set_xlabel(xlabel)

    ax.set_title(title, loc="left", fontweight="bold")

    ax.grid(axis="x", linestyle="--", alpha=0.18)





def create_main_figure(

    table: pd.DataFrame,

    model_name: str,

    n_test: int,

    n_events: int,

    n_bootstrap: int,

    output_dir: pathlib.Path,

) -> pathlib.Path:

    configure_style()

    colors = variable_colors()

    fig = plt.figure(figsize=(19.5, 15.8))

    grid = fig.add_gridspec(

        2,

        2,

        left=0.075,

        right=0.985,

        bottom=0.075,

        top=0.94,

        hspace=0.38,

        wspace=0.34,

    )

    fig.suptitle(

        "Subgroup performance and calibration of the hybrid DARN–XGBoost blend",

        fontsize=19,

        fontweight="bold",

        y=0.982,

    )

    # Cohort size, model choice, and threshold provenance are reported in the caption.

    ax_a = fig.add_subplot(grid[0, 0])

    lows = pd.to_numeric(table["AUROC_ci_lower_95"], errors="coerce").dropna()

    highs = pd.to_numeric(table["AUROC_ci_upper_95"], errors="coerce").dropna()

    auc_xlim = (

        max(0.0, float(lows.min()) - 0.04),

        min(1.0, float(highs.max()) + 0.025),

    ) if len(lows) and len(highs) else (0.5, 1.0)

    forest_panel(

        ax_a,

        table,

        "AUROC",

        "A. Discrimination across clinical subgroups",

        "AUROC with bootstrap 95% CI",

        colors,

        xlim=auc_xlim,

        show_ylabels=True,

    )



    ax_b = fig.add_subplot(grid[0, 1])

    pr_highs = pd.to_numeric(table["AUPRC_ci_upper_95"], errors="coerce").dropna()

    pr_upper = max(

        float(table["prevalence"].max()) * 1.25,

        float(pr_highs.max()) * 1.08 if len(pr_highs) else 0.1,

        0.05,

    )

    forest_panel(

        ax_b,

        table,

        "AUPRC",

        "B. Precision–recall performance across subgroups",

        "AUPRC with bootstrap 95% CI",

        colors,

        xlim=(0.0, min(1.0, pr_upper)),

        show_ylabels=False,

        prevalence_markers=True,

    )



    ax_c = fig.add_subplot(grid[1, 0])

    oe_highs = pd.to_numeric(

        table["observed_expected_ratio_ci_upper_95"], errors="coerce"

    ).replace([np.inf, -np.inf], np.nan).dropna()

    oe_upper = min(5.0, max(1.6, float(oe_highs.max()) * 1.08)) if len(oe_highs) else 2.0

    forest_panel(

        ax_c,

        table,

        "observed_expected_ratio",

        "C. Calibration-in-the-large across subgroups",

        "Observed / expected deaths with bootstrap 95% CI",

        colors,

        xlim=(0.0, oe_upper),

        reference=1.0,

        show_ylabels=True,

    )

    ax_c.text(1.0, len(table) - 0.55, "Ideal", ha="left", va="top", fontsize=11.0, color="dimgray")



    nested = grid[1, 1].subgridspec(1, 2, wspace=0.18)

    ax_d1 = fig.add_subplot(nested[0, 0])

    ax_d2 = fig.add_subplot(nested[0, 1], sharey=ax_d1)

    forest_panel(

        ax_d1,

        table,

        "sensitivity",

        "D. Operating-point behavior",

        "Sensitivity",

        colors,

        xlim=(0.0, 1.02),

        show_ylabels=True,

    )

    forest_panel(

        ax_d2,

        table,

        "FPR",

        "",

        "False-positive rate",

        colors,

        xlim=(0.0, 1.02),

        show_ylabels=False,

    )

    ax_d2.spines["left"].set_visible(False)



    legend = [

        Line2D([0], [0], marker="o", color=colors[name], linestyle="none", markersize=7, label=name)

        for name in VARIABLE_ORDER

    ]

    legend.extend(

        [

            Line2D([0], [0], marker="X", color="gray", linestyle="none", markersize=7, label=f"Exploratory: <{MIN_EVENTS_DISPLAY} deaths"),

            Line2D([0], [0], marker="|", color="dimgray", linestyle="none", markersize=10, markeredgewidth=1.8, label="Subgroup prevalence in Panel B"),

        ]

    )

    fig.legend(

        handles=legend,

        frameon=False,

        ncol=7,

        loc="lower center",

        bbox_to_anchor=(0.5, 0.018),

        fontsize=11.0,

    )

    # Exploratory-subgroup limitations are stated in the figure caption.

    path = output_dir / f"{FIGURE_BASENAME}.png"

    save_figure_all_formats(fig, path)

    plt.show()

    plt.close(fig)

    return path





def create_urgency_pr_figure(

    groups: Mapping[str, pd.Series],

    y: np.ndarray,

    p: np.ndarray,

    output_dir: pathlib.Path,

) -> Optional[pathlib.Path]:

    if "Urgency" not in groups:

        print("Urgency unavailable; supplementary PR figure skipped.")

        return None

    configure_style()

    urgency = groups["Urgency"].reset_index(drop=True)

    cmap = cm.get_cmap(CMAP_NAME)

    colors = {"Elective": cmap(0.30), "Emergency": cmap(0.82)}

    fig, ax = plt.subplots(figsize=(8.5, 6.8))

    plotted = False

    for level in ["Elective", "Emergency"]:

        mask = urgency.eq(level).to_numpy()

        yg, pg = y[mask], p[mask]

        if len(yg) == 0 or len(np.unique(yg)) < 2:

            continue

        precision, recall, _ = precision_recall_curve(yg, pg)

        auprc = average_precision_score(yg, pg)

        prevalence = float(yg.mean())

        ax.plot(

            recall,

            precision,

            linewidth=2.5,

            color=colors[level],

            label=(

                f"{level}: AUPRC={auprc:.3f}; prevalence={prevalence:.3%}; "

                f"N={len(yg):,}; deaths={int(yg.sum())}"

            ),

        )

        ax.axhline(prevalence, linewidth=1.0, linestyle=":", color=colors[level], alpha=0.65)

        plotted = True

    if not plotted:

        plt.close(fig)

        return None

    ax.set_xlim(0, 1)

    ax.set_ylim(bottom=0)

    ax.set_xlabel("Recall (sensitivity)")

    ax.set_ylabel("Precision (positive predictive value)")

    ax.set_title(

        "Supplementary: elective and emergency precision–recall curves",

        loc="left",

        fontweight="bold",

    )

    ax.grid(linestyle="--", alpha=0.18)

    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()

    path = output_dir / f"{SUPPLEMENTARY_BASENAME}.png"

    save_figure_all_formats(fig, path)

    plt.show()

    plt.close(fig)

    return path



# =============================================================================

# TEXT OUTPUT

# =============================================================================



def metric_ci_text(row: pd.Series, metric: str, digits: int = 3) -> str:

    estimate = float(row[metric])

    low = float(row[f"{metric}_ci_lower_95"])

    high = float(row[f"{metric}_ci_upper_95"])

    if not (np.isfinite(estimate) and np.isfinite(low) and np.isfinite(high)):

        return "not estimable"

    return f"{estimate:.{digits}f} (95% CI {low:.{digits}f}–{high:.{digits}f})"





def manuscript_text(

    table: pd.DataFrame,

    gaps: pd.DataFrame,

    model_name: str,

    threshold: float,

    n_test: int,

    n_events: int,

) -> str:

    lines = [

        "FIGURE 6 — TEMPORAL-SAFE STRICT-OOF SUBGROUP NUMERICAL RESULTS",

        "=" * 84,

        "",

        (

            f"The internal held-out test cohort included {n_test:,} encounters and "

            f"{n_events} deaths (prevalence {100*n_events/n_test:.3f}%)."

        ),

        (

            f"The subgroup audit used {model_name}, evaluated at the "

            f"development-defined threshold of {threshold:.8f}."

        ),

        (

            "Subgroup analyses are exploratory because several strata contain "

            "few deaths and have wide confidence intervals."

        ),

        "",

        "SUBGROUP RESULTS",

    ]

    for _, row in table.iterrows():

        lines.append(

            f"{row['display_label']}: N={int(row['n']):,}, deaths={int(row['events'])}; "

            f"AUROC {metric_ci_text(row, 'AUROC')}; "

            f"AUPRC {metric_ci_text(row, 'AUPRC')} versus prevalence {float(row['prevalence']):.4f}; "

            f"O/E {metric_ci_text(row, 'observed_expected_ratio')}; "

            f"sensitivity {metric_ci_text(row, 'sensitivity')}; "

            f"FPR {metric_ci_text(row, 'FPR')}."

        )

    lines.extend(["", "DESCRIPTIVE BETWEEN-STRATUM GAPS"])

    if gaps.empty:

        lines.append("No between-stratum gaps were estimable.")

    else:

        for variable in ["Sex", "Age", "ASA", "Urgency"]:

            subset = gaps.loc[

                gaps["subgroup_variable"].eq(variable)

                & gaps["metric"].isin(["AUROC", "AUPRC", "sensitivity", "FPR"])

            ]

            for row in subset.itertuples():

                lines.append(

                    f"{variable} {row.metric} range: {row.absolute_gap:.3f} "

                    f"({row.minimum_group}={row.minimum:.3f}; "

                    f"{row.maximum_group}={row.maximum:.3f})."

                )

    return "\n".join(lines)



# =============================================================================

# MAIN

# =============================================================================



def run_figure6(

    run_dir: pathlib.Path,

    pipeline_module: pathlib.Path,

    data_path: pathlib.Path,

    output_dir: pathlib.Path,

    primary_model: Optional[str],

    n_bootstrap: int,

    n_jobs: int,

) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)



    print("=" * 100)

    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 6")

    print("=" * 100)

    print("Run directory   :", run_dir)

    print("Pipeline module :", pipeline_module)
    print("Output directory:", output_dir)



    module = load_pipeline_module(pipeline_module)

    reconstructed = reconstruct_holdout(module, run_dir, data_path)



    # Focused subgroup audit of one reference model. The hybrid blend is
    # the default for continuity with the operating-point analyses; this
    # display choice is not a claim of model superiority.
    resolved_primary_model = (
        PRIMARY_MODEL
        if primary_model is None
        or str(primary_model).strip().lower() in {"", "auto"}
        else str(primary_model)
    )


    primary = load_primary_model(

        run_dir,

        reconstructed["predictions"],

        resolved_primary_model,

    )



    y = reconstructed["y_true"]

    p = primary["probability"]

    threshold = float(primary["threshold"])

    groups = derive_subgroups(reconstructed["holdout"])



    print("\n" + "=" * 100)

    print("FIGURE 6 — CURRENT STRICT-OOF SUBGROUP AUDIT")

    print("=" * 100)

    print("Run directory:", run_dir)

    print("Audit model:", resolved_primary_model)

    print("Probability column:", primary["probability_column"])

    print(f"Development-defined threshold: {threshold:.8f}")

    print(f"Test set: N={len(y):,}; deaths={int(y.sum())}")



    subgroup_table = calculate_subgroup_table(

        y,

        p,

        threshold,

        groups,

        n_bootstrap=n_bootstrap,

        n_jobs=n_jobs,

    )

    main_table = (

        subgroup_table.loc[subgroup_table["display_order"].lt(999)]

        .sort_values("display_order")

        .reset_index(drop=True)

    )

    if main_table.empty:

        raise RuntimeError("No prespecified main-figure subgroups were available.")



    gaps = calculate_gap_table(subgroup_table)

    urgency = (

        subgroup_table.loc[

            subgroup_table["subgroup_variable"].eq("Urgency")

            & subgroup_table["subgroup_level"].isin(["Elective", "Emergency"])

        ]

        .copy()

        .sort_values(

            "subgroup_level",

            key=lambda values: values.map({"Elective": 0, "Emergency": 1}),

        )

    )



    main_figure = create_main_figure(

        main_table,

        model_name=resolved_primary_model,

        n_test=len(y),

        n_events=int(y.sum()),

        n_bootstrap=n_bootstrap,

        output_dir=output_dir,

    )

    urgency_figure = create_urgency_pr_figure(groups, y, p, output_dir)



    subgroup_path = output_dir / "Figure6_subgroup_metrics_with_bootstrap_CIs.csv"

    gap_path = output_dir / "Figure6_subgroup_gap_summary.csv"

    urgency_path = output_dir / "Figure6_urgency_summary.csv"

    manuscript_path = output_dir / "Figure6_manuscript_numerics.txt"

    summary_path = output_dir / "Figure6_temporal_safe_strict_oof_summary.json"



    subgroup_table.to_csv(subgroup_path, index=False)

    gaps.to_csv(gap_path, index=False)

    urgency.to_csv(urgency_path, index=False)



    text = manuscript_text(

        main_table,

        gaps,

        model_name=resolved_primary_model,

        threshold=threshold,

        n_test=len(y),

        n_events=int(y.sum()),

    )

    manuscript_path.write_text(text, encoding="utf-8")



    overall = subgroup_table.loc[

        subgroup_table["subgroup_variable"].eq("Overall")

    ].iloc[0]

    threshold_transport_mode = str(

        reconstructed["config"].get(

            "threshold_transport_mode",

            "absolute",

        )

    )



    summary = {

        "analysis_version": "temporal-safe-strict-oof",

        "run_id": run_dir.name,

        "run_directory": str(run_dir),

        "pipeline_module": str(pipeline_module),

        "data_path": str(reconstructed["data_path"]),

        "subgroup_audit_model": resolved_primary_model,

        "probability_column": primary["probability_column"],

        "threshold": threshold,

        "threshold_source": (

            "Cross-fitted development predictions; transported to the locked "

            "holdout according to the saved run configuration and read from "

            "test_model_metrics.csv"

        ),

        "threshold_transport_mode": threshold_transport_mode,

        "threshold_label": THRESHOLD_LABEL,

        "test_n": int(len(y)),

        "test_events": int(y.sum()),

        "test_prevalence": float(y.mean()),

        "bootstrap_replicates_per_subgroup": int(n_bootstrap),

        "minimum_events_for_emphasis": int(MIN_EVENTS_DISPLAY),

        "main_panels": {

            "A": "AUROC across prespecified clinical subgroups",

            "B": "AUPRC across subgroups relative to subgroup-specific prevalence",

            "C": "Observed-to-expected mortality ratio across subgroups",

            "D": "Sensitivity and false-positive rate at the development-defined operating point",

        },

        "supplementary_panel": "Elective versus emergency precision-recall curves" if urgency_figure else None,

        "removed_old_panel": (

            "Three-seed validation stability is not shown because the "

            "strict-OOF analysis uses five outer cross-validation folds and "

            "fold-specific fitted models rather than a three-seed repeated "

            "training design."

        ),

        "overall_AUROC": float(overall["AUROC"]),

        "overall_AUPRC": float(overall["AUPRC"]),

        "overall_Brier": float(overall["Brier"]),

        "overall_observed_expected_ratio": float(overall["observed_expected_ratio"]),

        "overall_sensitivity": float(overall["sensitivity"]),

        "overall_FPR": float(overall["FPR"]),

        "main_figure": str(main_figure),

        "urgency_pr_figure": None if urgency_figure is None else str(urgency_figure),

        "interpretation_note": (

            "Exploratory internal subgroup audit of the locked strict-OOF "

            "held-out test cohort with 47 deaths. Wide confidence intervals and "

            "small-event strata preclude claims of demographic fairness, "

            "equalized performance, or external transportability."

        ),

    }

    save_json(summary, summary_path)



    print("\n" + "=" * 100)

    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 6 COMPLETE")

    print("=" * 100)

    print("Main figure:", main_figure)

    if urgency_figure:

        print("Supplementary urgency PR figure:", urgency_figure)

    print("Subgroup table:", subgroup_path)

    print("Gap table:", gap_path)

    print("Urgency table:", urgency_path)

    print("Summary:", summary_path)

    print("Manuscript numerics:", manuscript_path)



    print("\n" + "=" * 100)

    print("MAIN SUBGROUP RESULTS")

    print("=" * 100)

    print(

        main_table[

            [

                "display_label",

                "n",

                "events",

                "prevalence",

                "AUROC",

                "AUROC_ci_lower_95",

                "AUROC_ci_upper_95",

                "AUPRC",

                "AUPRC_ci_lower_95",

                "AUPRC_ci_upper_95",

                "observed_expected_ratio",

                "sensitivity",

                "FPR",

            ]

        ].round(6).to_string(index=False)

    )

    print("\n" + "=" * 100)

    print("MANUSCRIPT-READY NUMERICS")

    print("=" * 100)

    print(text)





def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        allow_abbrev=False,

        description="Generate the final temporal-safe strict-OOF 2x2 subgroup performance and calibration figure.",

    )

    parser.add_argument("--run_dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)

    parser.add_argument("--pipeline_module", type=pathlib.Path, default=DEFAULT_PIPELINE_MODULE)

    parser.add_argument("--data_path", type=pathlib.Path, default=DEFAULT_DATA_PATH)

    parser.add_argument("--output_dir", type=pathlib.Path, default=None)

    parser.add_argument(

        "--primary_model",

        type=str,

        default="auto",

        help=(

            "Model to audit. Default 'auto' uses Hybrid DARN-XGB Blend. "

            "This is a reference-model display choice, not a ranking."

        ),

    )

    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)

    parser.add_argument("--n_jobs", type=int, default=N_JOBS)



    raw = list(sys.argv[1:] if argv is None else argv)

    filtered: List[str] = []

    skip = False

    for argument in raw:

        if skip:

            skip = False

            continue

        if argument in {"-f", "--f"}:

            skip = True

            continue

        if argument.startswith("-f=") or argument.startswith("--f="):

            continue

        filtered.append(argument)

    args, unknown = parser.parse_known_args(filtered)

    if unknown:

        print("Ignoring unrecognized Jupyter/IPython arguments:", " ".join(unknown))

    return args





def main() -> None:

    args = parse_args()

    output_dir = (

        args.output_dir

        if args.output_dir is not None

        else args.run_dir / DEFAULT_OUTPUT_SUBDIR

    )

    run_figure6(

        run_dir=args.run_dir,

        pipeline_module=args.pipeline_module,

        data_path=args.data_path,

        output_dir=output_dir,

        primary_model=args.primary_model,

        n_bootstrap=args.n_bootstrap,

        n_jobs=args.n_jobs,

    )





if __name__ == "__main__":

    main()