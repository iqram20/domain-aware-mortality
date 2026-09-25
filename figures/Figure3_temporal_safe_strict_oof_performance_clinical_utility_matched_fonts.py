#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

Standalone manuscript Figure 3 for the final temporal-safe, strict-OOF INSPIRE mortality run.



Panels

------

A. Comparative AUROC and AUPRC forest plots with stratified-bootstrap 95% confidence intervals

B. Receiver operating characteristic curves for all models

C. Sensitivity, specificity, balanced accuracy, and accuracy at the

   development-defined high-sensitivity operating point

D. Alert burden, PPV, and deaths detected at the same operating point

E. Calibration of the main models, with bootstrap CI for the hybrid blend

F. Decision-curve analysis over clinically plausible threshold probabilities



This script does not train, refit, or recalibrate any model. It reads the saved

outputs from Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py. The completed

figure is displayed inline by default, and detailed model, operating-point,

and calibration results are printed to the console.

"""



from __future__ import annotations



import argparse

import json

import math

import pathlib

import re

import warnings

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

import numpy as np

import pandas as pd

from sklearn.metrics import (

    average_precision_score,

    brier_score_loss,

    precision_recall_curve,

    roc_auc_score,

    roc_curve,

)



warnings.filterwarnings("ignore")





# =============================================================================

# DEFAULT SETTINGS FOR THE CURRENT CORRECTED RUN

# =============================================================================



DEFAULT_RUN_DIR = pathlib.Path(

    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"

    "darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653"

)



MODEL_ORDER = [

    "Hybrid DARN-XGB Blend",

    "Hybrid DARN-XGB Stack",

    "XGBoost",

    "Uniform DARN",

    "MLP",

    "Logistic Regression",

    "ASA-only Logistic",

]



DISPLAY_NAMES = {

    "Hybrid DARN-XGB Blend": "Hybrid blend",

    "Hybrid DARN-XGB Stack": "Hybrid stack",

    "XGBoost": "XGBoost",

    "Uniform DARN": "Uniform DARN",

    "MLP": "Standard MLP",

    "Logistic Regression": "Logistic regression",

    "ASA-only Logistic": "ASA-only logistic",

}



REFERENCE_CALIBRATION_MODEL = "Hybrid DARN-XGB Blend"

MAIN_CALIBRATION_MODELS = [

    "Hybrid DARN-XGB Blend",

    "Hybrid DARN-XGB Stack",

    "XGBoost",

    "Uniform DARN",

]

MAIN_DCA_MODELS = MAIN_CALIBRATION_MODELS + ["Treat all", "Treat none"]



CMAP_NAME = "plasma"

RANDOM_STATE = 20260919

CALIBRATION_BOOTSTRAPS = 500

CALIBRATION_BINS = 10

FIGURE_BASENAME = "Figure3_TemporalSafe_StrictOOF_Performance_Clinical_Utility"

SHOW_FIGURE = True





# =============================================================================

# UTILITIES

# =============================================================================





def require_file(path: pathlib.Path) -> pathlib.Path:

    if not path.exists():

        raise FileNotFoundError(

            f"Required file was not found:\n{path}\n"

            "Confirm that --run_dir points to the completed temporal-safe strict-OOF base run."

        )

    return path





def safe_name(name: str) -> str:

    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()





def probability_column(model_name: str) -> str:

    return f"{safe_name(model_name)}_calibrated_probability"





def load_run_config(run_dir: pathlib.Path) -> Dict[str, object]:

    """Load the exact configuration saved by the completed strict-OOF run."""

    config_path = run_dir / "config.json"

    if not config_path.exists():

        return {}

    with open(config_path, "r", encoding="utf-8") as handle:

        return json.load(handle)





def percentile_ci(values: np.ndarray) -> Tuple[float, float]:

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if values.size == 0:

        return np.nan, np.nan

    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))





def stratified_bootstrap_indices(

    y_true: np.ndarray,

    rng: np.random.Generator,

) -> np.ndarray:

    positives = np.where(y_true == 1)[0]

    negatives = np.where(y_true == 0)[0]

    sampled_positive = rng.choice(positives, size=len(positives), replace=True)

    sampled_negative = rng.choice(negatives, size=len(negatives), replace=True)

    sampled = np.concatenate([sampled_positive, sampled_negative])

    rng.shuffle(sampled)

    return sampled





def calibration_table(

    y_true: np.ndarray,

    probabilities: np.ndarray,

    n_bins: int,

) -> pd.DataFrame:

    """Quantile calibration bins defined on the complete test predictions."""

    y = np.asarray(y_true, dtype=int)

    p = np.asarray(probabilities, dtype=float)

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))

    if len(edges) < 3:

        raise ValueError("Insufficient unique probabilities for calibration bins.")

    bin_ids = np.digitize(p, edges[1:-1], right=True)

    rows: List[Dict[str, float]] = []

    for bin_id in range(len(edges) - 1):

        mask = bin_ids == bin_id

        if not mask.any():

            continue

        rows.append(

            {

                "bin": int(bin_id),

                "n": int(mask.sum()),

                "events": int(y[mask].sum()),

                "mean_predicted": float(p[mask].mean()),

                "observed_rate": float(y[mask].mean()),

            }

        )

    return pd.DataFrame(rows)





def calibration_table_with_bootstrap_ci(

    y_true: np.ndarray,

    probabilities: np.ndarray,

    n_bins: int,

    n_bootstrap: int,

    random_state: int,

) -> pd.DataFrame:

    """Calibration table with stratified-bootstrap CIs using fixed test-bin edges."""

    y = np.asarray(y_true, dtype=int)

    p = np.asarray(probabilities, dtype=float)

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))

    if len(edges) < 3:

        raise ValueError("Insufficient unique probabilities for calibration bins.")



    bin_ids = np.digitize(p, edges[1:-1], right=True)

    observed_boot: List[List[float]] = [[] for _ in range(len(edges) - 1)]

    rng = np.random.default_rng(random_state)



    for _ in range(n_bootstrap):

        indices = stratified_bootstrap_indices(y, rng)

        y_boot = y[indices]

        p_boot = p[indices]

        boot_bin_ids = np.digitize(p_boot, edges[1:-1], right=True)

        for bin_id in range(len(edges) - 1):

            mask = boot_bin_ids == bin_id

            if mask.any():

                observed_boot[bin_id].append(float(y_boot[mask].mean()))



    rows: List[Dict[str, float]] = []

    for bin_id in range(len(edges) - 1):

        mask = bin_ids == bin_id

        if not mask.any():

            continue

        low, high = percentile_ci(np.asarray(observed_boot[bin_id], dtype=float))

        rows.append(

            {

                "bin": int(bin_id),

                "n": int(mask.sum()),

                "events": int(y[mask].sum()),

                "mean_predicted": float(p[mask].mean()),

                "observed_rate": float(y[mask].mean()),

                "CI_low": low,

                "CI_high": high,

            }

        )

    return pd.DataFrame(rows)





def load_metrics_with_ci(

    metrics_file: pathlib.Path,

    bootstrap_file: pathlib.Path,

    model_order: Sequence[str],

) -> pd.DataFrame:

    metrics = pd.read_csv(require_file(metrics_file))

    bootstrap = pd.read_csv(require_file(bootstrap_file))



    required_metric_columns = {

        "model", "AUROC", "AUPRC", "Brier", "Accuracy",

        "Balanced_Accuracy", "Recall", "Specificity", "Precision",

        "TP", "FP", "FN", "TN",

    }

    missing = required_metric_columns - set(metrics.columns)

    if missing:

        raise KeyError(f"test_model_metrics.csv is missing: {sorted(missing)}")



    pivot_low = bootstrap.pivot(index="model", columns="metric", values="CI_low")

    pivot_high = bootstrap.pivot(index="model", columns="metric", values="CI_high")



    for metric in [

        "AUROC", "AUPRC", "Brier", "Accuracy", "Balanced_Accuracy",

        "Recall", "Specificity", "Precision",

    ]:

        if metric in pivot_low.columns:

            metrics[f"{metric}_CI_low"] = metrics["model"].map(pivot_low[metric])

            metrics[f"{metric}_CI_high"] = metrics["model"].map(pivot_high[metric])



    available = [name for name in model_order if name in set(metrics["model"])]

    if not available:

        raise ValueError("None of the expected final-run models were found.")



    return (

        metrics.set_index("model")

        .loc[available]

        .reset_index()

    )





def set_figure_style() -> None:

    mpl.rcParams.update(

        {

            "font.family": "DejaVu Sans",

            "font.size": 12,

            "axes.titlesize": 14,

            "axes.labelsize": 12.5,

            "xtick.labelsize": 11.5,

            "ytick.labelsize": 11.5,

            "legend.fontsize": 11,

            "axes.linewidth": 0.9,

            "axes.spines.top": False,

            "axes.spines.right": False,

            "figure.facecolor": "white",

            "savefig.facecolor": "white",

            "pdf.fonttype": 42,

            "ps.fonttype": 42,

        }

    )





def create_model_colors(model_order: Sequence[str]) -> Dict[str, Tuple[float, ...]]:

    cmap = cm.get_cmap(CMAP_NAME)

    positions = np.linspace(0.08, 0.90, len(model_order))

    return {name: cmap(position) for name, position in zip(model_order, positions)}





def model_line_style(model_name: str) -> str:

    return {

        "Hybrid DARN-XGB Blend": "-",

        "Hybrid DARN-XGB Stack": "--",

        "XGBoost": "-.",

        "Uniform DARN": ":",

        "Treat all": "--",

        "Treat none": ":",

    }.get(model_name, "-")





def model_line_width(model_name: str) -> float:

    if model_name in {"Hybrid DARN-XGB Blend", "Hybrid DARN-XGB Stack"}:

        return 2.8

    if model_name in {"XGBoost", "Uniform DARN"}:

        return 2.2

    return 1.4





# =============================================================================

# FIGURE GENERATION

# =============================================================================





def build_figure(

    run_dir: pathlib.Path,

    output_dir: pathlib.Path,

    calibration_bootstraps: int,

    *,

    show_figure: bool = SHOW_FIGURE,

) -> Dict[str, pathlib.Path]:

    print("=" * 100)

    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 3")

    print("=" * 100)

    print("Run directory   :", run_dir)

    print("Output directory:", output_dir)

    print("Inline display  :", bool(show_figure))

    print()



    prediction_file = run_dir / "test_predictions_all_models.csv"

    metrics_file = run_dir / "test_model_metrics.csv"

    bootstrap_file = run_dir / "bootstrap_confidence_intervals.csv"

    decision_curve_file = run_dir / "decision_curve_analysis.csv"

    paired_hybrid_file = run_dir / "paired_bootstrap_hybrid_vs_all.csv"



    run_config = load_run_config(run_dir)

    predictions = pd.read_csv(require_file(prediction_file))

    metrics = load_metrics_with_ci(metrics_file, bootstrap_file, MODEL_ORDER)

    decision_curve = pd.read_csv(require_file(decision_curve_file))



    if "y_true" not in predictions.columns:

        raise KeyError("test_predictions_all_models.csv does not contain y_true.")



    y_true = predictions["y_true"].astype(int).to_numpy()

    n_test = int(len(y_true))

    n_events = int(y_true.sum())

    prevalence = float(y_true.mean())



    available_models = metrics["model"].astype(str).tolist()



    configured_reference = REFERENCE_CALIBRATION_MODEL

    reference_model = (
        configured_reference
        if configured_reference in available_models
        else available_models[0]
    )


    calibration_models = list(

        dict.fromkeys(

            [

                reference_model,

                "Hybrid DARN-XGB Stack",

                "XGBoost",

                "Uniform DARN",

            ]

        )

    )

    calibration_models = [

        model for model in calibration_models if model in available_models

    ]

    dca_models = calibration_models + ["Treat all", "Treat none"]



    model_probabilities: Dict[str, np.ndarray] = {}

    for model_name in available_models:

        column = probability_column(model_name)

        if column not in predictions.columns:

            raise KeyError(f"Missing calibrated probability column: {column}")

        model_probabilities[model_name] = predictions[column].astype(float).to_numpy()



    model_colors = create_model_colors(available_models)

    output_dir.mkdir(parents=True, exist_ok=True)

    set_figure_style()



    # Numerical consistency check against the saved table.

    for model_name, p in model_probabilities.items():

        saved = metrics.loc[metrics["model"] == model_name].iloc[0]

        computed_auc = roc_auc_score(y_true, p)

        computed_ap = average_precision_score(y_true, p)

        if not np.isclose(computed_auc, saved["AUROC"], atol=1e-6):

            raise ValueError(f"AUROC mismatch for {model_name}.")

        if not np.isclose(computed_ap, saved["AUPRC"], atol=1e-6):

            raise ValueError(f"AUPRC mismatch for {model_name}.")



    # Calibration tables.

    calibration_tables: List[pd.DataFrame] = []

    for model_name in calibration_models:

        if model_name not in model_probabilities:

            continue

        if model_name == reference_model:

            table = calibration_table_with_bootstrap_ci(

                y_true,

                model_probabilities[model_name],

                n_bins=CALIBRATION_BINS,

                n_bootstrap=calibration_bootstraps,

                random_state=RANDOM_STATE,

            )

            table["has_CI"] = True

        else:

            table = calibration_table(

                y_true,

                model_probabilities[model_name],

                n_bins=CALIBRATION_BINS,

            )

            table["CI_low"] = np.nan

            table["CI_high"] = np.nan

            table["has_CI"] = False

        table["model"] = model_name

        calibration_tables.append(table)

    calibration_values = pd.concat(calibration_tables, ignore_index=True)



    # Main canvas.

    fig = plt.figure(figsize=(18, 19.2))

    grid = fig.add_gridspec(

        3,

        2,

        left=0.075,

        right=0.98,

        bottom=0.055,

        top=0.945,

        hspace=0.38,

        wspace=0.32,

    )



    fig.suptitle(

        "Model performance and clinical utility for 30-day postoperative mortality prediction",

        fontsize=18,

        fontweight="bold",

        y=0.982,

    )

    # Run provenance and threshold-selection details are reported in the manuscript caption.

    # -------------------------------------------------------------------------

    # A. Comparative AUROC and AUPRC forest plots

    # -------------------------------------------------------------------------

    forest = metrics.iloc[::-1].reset_index(drop=True)

    y_positions = np.arange(len(forest))



    panel_a_grid = grid[0, 0].subgridspec(

        1,

        2,

        width_ratios=[1.0, 1.0],

        wspace=0.22,

    )

    ax_a_auc = fig.add_subplot(panel_a_grid[0, 0])

    ax_a_pr = fig.add_subplot(panel_a_grid[0, 1], sharey=ax_a_auc)



    for i, row in forest.iterrows():

        name = row["model"]

        emphasized = name in {

            "Hybrid DARN-XGB Blend",

            "Hybrid DARN-XGB Stack",

            "XGBoost",

        }

        marker_size = 6.8 if emphasized else 5.4

        line_width = 1.7 if emphasized else 1.15



        auc_low_ci = row.get("AUROC_CI_low", np.nan)

        auc_high_ci = row.get("AUROC_CI_high", np.nan)

        auc_error = None

        if np.isfinite(auc_low_ci) and np.isfinite(auc_high_ci):

            auc_error = np.array(

                [[row["AUROC"] - auc_low_ci], [auc_high_ci - row["AUROC"]]]

            )

        ax_a_auc.errorbar(

            row["AUROC"],

            i,

            xerr=auc_error,

            fmt="o",

            markersize=marker_size,

            capsize=2.5,

            linewidth=line_width,

            color=model_colors[name],

            markeredgecolor="white",

            markeredgewidth=0.55,

            zorder=3,

        )



        pr_low_ci = row.get("AUPRC_CI_low", np.nan)

        pr_high_ci = row.get("AUPRC_CI_high", np.nan)

        pr_error = None

        if np.isfinite(pr_low_ci) and np.isfinite(pr_high_ci):

            pr_error = np.array(

                [[row["AUPRC"] - pr_low_ci], [pr_high_ci - row["AUPRC"]]]

            )

        ax_a_pr.errorbar(

            row["AUPRC"],

            i,

            xerr=pr_error,

            fmt="o",

            markersize=marker_size,

            capsize=2.5,

            linewidth=line_width,

            color=model_colors[name],
            markeredgecolor="white",

            markeredgewidth=0.55,

            zorder=3,

        )



    auc_low = max(0.0, float(forest["AUROC_CI_low"].min()) - 0.025)

    auc_high = min(1.005, float(forest["AUROC_CI_high"].max()) + 0.012)

    pr_high = min(

        1.0,

        max(0.12, float(forest["AUPRC_CI_high"].max()) * 1.10),

    )



    ax_a_auc.set_xlim(auc_low, auc_high)

    ax_a_pr.set_xlim(0.0, pr_high)

    ax_a_auc.set_ylim(-0.55, len(forest) - 0.45)



    ax_a_auc.set_yticks(y_positions)

    ax_a_auc.set_yticklabels(

        [DISPLAY_NAMES.get(name, name) for name in forest["model"]],

        fontsize=11.5,

    )

    ax_a_pr.tick_params(axis="y", left=False, labelleft=False)

    ax_a_pr.spines["left"].set_visible(False)



    ax_a_auc.set_xlabel("AUROC (95% CI)")

    ax_a_pr.set_xlabel("AUPRC (95% CI)")

    ax_a_auc.set_title(

        "A. Comparative predictive performance",

        loc="left",

        fontweight="bold",

        pad=10,

    )



    ax_a_pr.axvline(

        prevalence,

        linestyle=":",

        linewidth=1.2,

        color="gray",

        zorder=1,

    )

    ax_a_pr.text(

        prevalence,

        len(forest) - 0.45,

        f" prevalence\n{prevalence:.4f}",

        rotation=90,

        va="top",

        ha="left",

        fontsize=10.5,

        color="dimgray",

    )



    for axis in (ax_a_auc, ax_a_pr):

        axis.grid(axis="x", linestyle="--", alpha=0.18)

        axis.tick_params(axis="x", labelsize=8.0)



    # -------------------------------------------------------------------------

    # B. Receiver operating characteristic curves

    # -------------------------------------------------------------------------

    ax_b = fig.add_subplot(grid[0, 1])



    for model_name in available_models:

        p = model_probabilities[model_name]

        fpr, tpr, _ = roc_curve(y_true, p)

        auc_value = float(

            metrics.loc[metrics["model"] == model_name, "AUROC"].iloc[0]

        )



        if model_name == "Hybrid DARN-XGB Blend":

            line_width = 3.0

            line_style = "-"

        elif model_name == "Hybrid DARN-XGB Stack":

            line_width = 2.7

            line_style = "--"

        elif model_name == "XGBoost":

            line_width = 2.4

            line_style = "-."

        elif model_name == "Uniform DARN":

            line_width = 2.0

            line_style = "-"

        elif "Logistic" in model_name:

            line_width = 1.4

            line_style = ":"

        else:

            line_width = 1.6

            line_style = "-"



        ax_b.plot(

            fpr,

            tpr,

            color=model_colors[model_name],

            linewidth=line_width,

            linestyle=line_style,

            label=f"{DISPLAY_NAMES.get(model_name, model_name)} ({auc_value:.3f})",

        )



    ax_b.plot(

        [0.0, 1.0],

        [0.0, 1.0],

        linestyle=":",

        linewidth=1.0,

        color="gray",

        label="Chance",

    )

    ax_b.set_xlim(0.0, 1.0)

    ax_b.set_ylim(0.0, 1.01)

    ax_b.set_xlabel("False-positive rate")

    ax_b.set_ylabel("True-positive rate")

    ax_b.set_title(

        "B. Receiver operating characteristic curves",

        loc="left",

        fontweight="bold",

    )

    ax_b.grid(linestyle="--", alpha=0.20)

    ax_b.legend(

        frameon=False,

        fontsize=10.5,

        loc="lower right",

    )



    # -------------------------------------------------------------------------

    # C. Threshold-dependent performance

    # -------------------------------------------------------------------------

    ax_c = fig.add_subplot(grid[1, 0])

    x = np.arange(len(metrics))

    width = 0.24

    bar_metrics = [

        ("Recall", "Sensitivity", -width),

        ("Specificity", "Specificity", 0.0),

        ("Balanced_Accuracy", "Balanced accuracy", width),

    ]

    metric_shades = [0.18, 0.50, 0.80]

    cmap = cm.get_cmap(CMAP_NAME)



    for (column, label, offset), shade in zip(bar_metrics, metric_shades):

        ax_c.bar(

            x + offset,

            metrics[column].astype(float),

            width=width,

            color=cmap(shade),

            alpha=0.88,

            label=label,

            edgecolor="none",

        )



    ax_c.scatter(

        x,

        metrics["Accuracy"].astype(float),

        marker="D",

        s=38,

        facecolor="white",

        edgecolor="black",

        linewidth=1.0,

        label="Accuracy",

        zorder=4,

    )

    ax_c.set_xticks(x)

    ax_c.set_xticklabels(

        [DISPLAY_NAMES.get(name, name) for name in metrics["model"]],

        rotation=27,

        ha="right",

        fontsize=11.5,

    )

    ax_c.set_ylim(0, 1.05)

    ax_c.set_ylabel("Performance")

    ax_c.set_title("C. Performance at the development-defined operating point", loc="left", fontweight="bold")

    ax_c.grid(axis="y", linestyle="--", alpha=0.20)

    ax_c.legend(frameon=False, fontsize=11, ncol=2, loc="upper left")



    # -------------------------------------------------------------------------

    # D. Alert burden and PPV

    # -------------------------------------------------------------------------

    ax_d = fig.add_subplot(grid[1, 1])

    ax_d_ppv = ax_d.twinx()

    alerts_per_1000 = 1000.0 * (metrics["TP"] + metrics["FP"]) / (

        metrics["TP"] + metrics["FP"] + metrics["FN"] + metrics["TN"]

    )

    bars = ax_d.bar(

        x,

        alerts_per_1000,

        width=0.66,

        color=[model_colors[name] for name in metrics["model"]],

        alpha=0.76,

        edgecolor="none",

        label="Alerts per 1,000",

    )

    ax_d_ppv.plot(

        x,

        100.0 * metrics["Precision"].astype(float),

        color="black",

        marker="o",

        linewidth=2.1,

        markersize=5.5,

        label="PPV",

    )



    max_alerts = float(alerts_per_1000.max())

    ax_d.set_ylim(0.0, max(10.0, 1.18 * max_alerts))

    for i, row in metrics.reset_index(drop=True).iterrows():

        ax_d.text(

            i,

            float(alerts_per_1000.iloc[i]) + max(4.0, 0.022 * max_alerts),

            f"{int(row['TP'])}/{n_events}",

            ha="center",

            va="bottom",

            fontsize=11.5,

            fontweight="bold" if "Hybrid" in row["model"] else "normal",

        )



    ax_d.set_xticks(x)

    ax_d.set_xticklabels(

        [DISPLAY_NAMES.get(name, name) for name in metrics["model"]],

        rotation=27,

        ha="right",

        fontsize=11.5,

    )

    ax_d.set_ylabel("Alerts per 1,000 procedures")

    ax_d_ppv.set_ylabel("Positive predictive value (%)")

    ax_d_ppv.set_ylim(0.0, max(5.0, 1.25 * 100.0 * float(metrics["Precision"].max())))

    ax_d.set_title("D. Clinical efficiency and alert burden", loc="left", fontweight="bold")

    ax_d.grid(axis="y", linestyle="--", alpha=0.20)

    handles = [

        Patch(facecolor=cmap(0.55), alpha=0.76, label="Alerts per 1,000"),

        Line2D([0], [0], color="black", marker="o", label="PPV"),

    ]

    ax_d.legend(handles=handles, frameon=False, fontsize=11.5, loc="upper left")

    ax_d.text(

        0.99,

        0.97,

        f"Numbers above bars indicate deaths detected (out of {n_events})",

        transform=ax_d.transAxes,

        ha="right",

        va="top",

        fontsize=10.5,

        color="dimgray",

    )



    # -------------------------------------------------------------------------

    # E. Calibration

    # -------------------------------------------------------------------------

    ax_e = fig.add_subplot(grid[2, 0])

    calibration_max_values = [0.01]

    for model_name in calibration_models:

        if model_name not in set(calibration_values["model"]):

            continue

        table = calibration_values.loc[calibration_values["model"] == model_name].sort_values("mean_predicted")

        x_cal = table["mean_predicted"].to_numpy(float)

        y_cal = table["observed_rate"].to_numpy(float)

        calibration_max_values.extend(x_cal[np.isfinite(x_cal)].tolist())

        calibration_max_values.extend(y_cal[np.isfinite(y_cal)].tolist())



        if model_name == reference_model:

            low = table["CI_low"].to_numpy(float)

            high = table["CI_high"].to_numpy(float)

            ax_e.fill_between(

                x_cal,

                low,

                high,

                color=model_colors[model_name],

                alpha=0.16,

                linewidth=0,

                label="Hybrid blend 95% bootstrap CI",

            )

        ax_e.plot(

            x_cal,

            y_cal,

            marker="o",

            markersize=4.8,

            linewidth=model_line_width(model_name),

            linestyle=model_line_style(model_name),

            color=model_colors[model_name],

            label=f"{DISPLAY_NAMES.get(model_name, model_name)} (Brier={float(metrics.loc[metrics['model'] == model_name, 'Brier'].iloc[0]):.6f})",

        )



    calibration_max = max(0.01, 1.12 * max(calibration_max_values))

    calibration_max = min(calibration_max, 0.08)

    ax_e.plot(

        [0, calibration_max],

        [0, calibration_max],

        color="gray",

        linestyle=":",

        linewidth=1.2,

        label="Ideal calibration",

    )

    ax_e.set_xlim(0, calibration_max)

    ax_e.set_ylim(0, calibration_max)

    ax_e.set_xlabel("Mean predicted mortality probability")

    ax_e.set_ylabel("Observed mortality rate")

    ax_e.set_title("E. Calibration of selected models", loc="left", fontweight="bold")

    ax_e.grid(linestyle="--", alpha=0.20)

    ax_e.legend(frameon=False, fontsize=10.5, loc="upper left")



    # -------------------------------------------------------------------------

    # F. Decision curve

    # -------------------------------------------------------------------------

    ax_f = fig.add_subplot(grid[2, 1])

    dca_colors = dict(model_colors)

    dca_colors["Treat all"] = "gray"

    dca_colors["Treat none"] = "black"



    dca_filtered = decision_curve.loc[

        decision_curve["model"].isin(dca_models)

    ].copy()

    for model_name in dca_models:

        table = dca_filtered.loc[dca_filtered["model"] == model_name].sort_values("threshold_probability")

        if table.empty:

            continue

        ax_f.plot(

            100.0 * table["threshold_probability"].astype(float),

            table["net_benefit"].astype(float),

            color=dca_colors[model_name],

            linestyle=model_line_style(model_name),

            linewidth=model_line_width(model_name),

            label=DISPLAY_NAMES.get(model_name, model_name),

        )



    x_min = 100.0 * float(dca_filtered["threshold_probability"].min())

    x_max = 100.0 * float(dca_filtered["threshold_probability"].max())

    ax_f.set_xscale("log")

    ax_f.set_xlim(x_min, x_max)

    ax_f.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)

    ax_f.set_xlabel("Threshold probability (%)")

    ax_f.set_ylabel("Net benefit")

    ax_f.set_title("F. Decision-curve analysis", loc="left", fontweight="bold")

    ax_f.grid(linestyle="--", alpha=0.20)

    ax_f.legend(frameon=False, fontsize=10.5, loc="best")



    # Save figure.

    png_path = output_dir / f"{FIGURE_BASENAME}.png"

    pdf_path = output_dir / f"{FIGURE_BASENAME}.pdf"

    svg_path = output_dir / f"{FIGURE_BASENAME}.svg"

    tiff_path = output_dir / f"{FIGURE_BASENAME}.tiff"



    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    fig.savefig(pdf_path, bbox_inches="tight")

    fig.savefig(svg_path, bbox_inches="tight")

    try:

        fig.savefig(

            tiff_path,

            dpi=600,

            bbox_inches="tight",

            pil_kwargs={"compression": "tiff_lzw"},

        )

    except Exception as exc:

        print(f"TIFF export skipped: {exc}")



    if show_figure:

        print("\nDisplaying Figure 3 inline...")

        plt.show()



    plt.close(fig)



    # Save figure-specific numerical outputs.

    metrics_out = output_dir / "Figure3_model_metrics_with_bootstrap_CIs.csv"

    calibration_out = output_dir / "Figure3_calibration_values.csv"

    operating_out = output_dir / "Figure3_operating_point_summary.csv"

    dca_out = output_dir / "Figure3_decision_curve_values.csv"

    paired_out = output_dir / "Figure3_paired_bootstrap_hybrid_vs_all.csv"

    summary_out = output_dir / "Figure3_summary.json"

    numerics_out = output_dir / "Figure3_manuscript_numerics.txt"

    provenance_out = output_dir / "Figure3_run_provenance.json"



    metrics.to_csv(metrics_out, index=False)

    calibration_values.to_csv(calibration_out, index=False)

    decision_curve.to_csv(dca_out, index=False)



    operating_summary = metrics[

        [

            "model", "Threshold", "Accuracy", "Balanced_Accuracy", "Recall",

            "Specificity", "Precision", "TP", "FP", "FN", "TN",

        ]

    ].copy()

    operating_summary["Alerts_per_1000"] = alerts_per_1000.to_numpy(float)

    operating_summary["Deaths_detected_out_of_total"] = (

        operating_summary["TP"].astype(int).astype(str) + f"/{n_events}"

    )

    operating_summary.to_csv(operating_out, index=False)



    if paired_hybrid_file.exists():

        pd.read_csv(paired_hybrid_file).to_csv(paired_out, index=False)



    reference_row = metrics.loc[

        metrics["model"] == reference_model

    ].iloc[0]



    stack_model = (

        "Hybrid DARN-XGB Stack"

        if "Hybrid DARN-XGB Stack" in available_models

        else reference_model

    )

    stack_row = metrics.loc[

        metrics["model"] == stack_model

    ].iloc[0]



    xgb_model = (

        "XGBoost"

        if "XGBoost" in available_models

        else reference_model

    )

    xgb_row = metrics.loc[

        metrics["model"] == xgb_model

    ].iloc[0]



    summary = {

        "run_directory": str(run_dir),

        "pipeline": "Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py",

        "test_n": n_test,

        "test_events": n_events,

        "prevalence": prevalence,

        "reference_model": reference_model,

        "reference_model_by_observed_AUPRC": reference_model,

        "reference_AUROC": float(reference_row["AUROC"]),

        "reference_AUPRC": float(reference_row["AUPRC"]),

        "reference_Brier": float(reference_row["Brier"]),

        "hybrid_stack_sensitivity": float(stack_row["Recall"]),

        "hybrid_stack_specificity": float(stack_row["Specificity"]),

        "hybrid_stack_alert_rate": float((stack_row["TP"] + stack_row["FP"]) / n_test),

        "xgboost_AUPRC": float(xgb_row["AUPRC"]),

        "calibration_bootstraps": int(calibration_bootstraps),

    }

    with open(summary_out, "w", encoding="utf-8") as handle:

        json.dump(summary, handle, indent=2)



    provenance = {

        "run_directory": str(run_dir),

        "figure_script": pathlib.Path(

            globals().get(

                "__file__",

                "Figure3_temporal_safe_strict_oof_performance_clinical_utility.py",

            )

        ).name,

        "figure_basename": FIGURE_BASENAME,

        "reference_model": reference_model,

        "available_models": available_models,

        "calibration_models": calibration_models,

        "decision_curve_models": dca_models,

        "saved_run_config": run_config,

    }

    with open(provenance_out, "w", encoding="utf-8") as handle:

        json.dump(provenance, handle, indent=2)



    lines = [

        "FIGURE 3 — TEMPORAL-SAFE STRICT-OOF MANUSCRIPT-READY NUMERICAL RESULTS",

        "=" * 78,

        "",

        (

            f"The held-out test cohort contained {n_test:,} encounters and "

            f"{n_events} deaths (prevalence {100.0 * prevalence:.3f}%)."

        ),

        "",

        (

            f"The {DISPLAY_NAMES.get(reference_model, reference_model)} had an observed "

            f"AUPRC of {reference_row['AUPRC']:.3f}, AUROC of "

            f"{reference_row['AUROC']:.3f}, and Brier score of "

            f"{reference_row['Brier']:.6f}."

        ),

        (

            f"The cross-fitted hybrid stack had sensitivity {stack_row['Recall']:.3f}, "

            f"specificity {stack_row['Specificity']:.3f}, PPV {stack_row['Precision']:.3f}, "

            f"and flagged {100.0 * (stack_row['TP'] + stack_row['FP']) / n_test:.1f}% of encounters."

        ),

        (

            f"{DISPLAY_NAMES.get(xgb_model, xgb_model)} had AUROC "

            f"{xgb_row['AUROC']:.3f} and AUPRC {xgb_row['AUPRC']:.3f}."

        ),

        "",

        "Model-specific point estimates and bootstrap confidence intervals are saved in:",

        str(metrics_out),

    ]

    with open(numerics_out, "w", encoding="utf-8") as handle:

        handle.write("\n".join(lines))



    print("=" * 100)

    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 3 COMPLETED")

    print("=" * 100)

    print(f"Test N: {n_test:,}")

    print(f"Deaths: {n_events}")

    print(f"Prevalence: {prevalence:.6f}")

    print("\nMODEL PERFORMANCE")

    print("-" * 100)

    print(

        metrics[

            [

                "model", "AUROC", "AUROC_CI_low", "AUROC_CI_high",

                "AUPRC", "AUPRC_CI_low", "AUPRC_CI_high", "Brier",

                "Accuracy", "Balanced_Accuracy", "Recall", "Specificity", "Precision",

            ]

        ].round(6).to_string(index=False)

    )



    print("\nOPERATING-POINT SUMMARY")

    print("-" * 100)

    console_operating = operating_summary[

        [

            "model",

            "Threshold",

            "TP",

            "FN",

            "FP",

            "TN",

            "Recall",
            "Specificity",

            "Precision",

            "Alerts_per_1000",

            "Deaths_detected_out_of_total",

        ]

    ].copy()

    console_operating = console_operating.rename(

        columns={

            "Recall": "Sensitivity",

            "Precision": "PPV",

        }

    )

    print(

        console_operating.round(

            {

                "Threshold": 8,

                "Sensitivity": 4,

                "Specificity": 4,

                "PPV": 4,

                "Alerts_per_1000": 2,

            }

        ).to_string(index=False)

    )



    reference_alerts = float(

        1000.0

        * (reference_row["TP"] + reference_row["FP"])

        / n_test

    )

    reference_flagged_percent = float(

        100.0

        * (reference_row["TP"] + reference_row["FP"])

        / n_test

    )



    print("\nREFERENCE HYBRID MODEL")

    print("-" * 100)

    print("Model:", reference_model)

    print(f"AUROC: {float(reference_row['AUROC']):.6f}")

    print(f"AUPRC: {float(reference_row['AUPRC']):.6f}")

    print(f"Brier: {float(reference_row['Brier']):.6f}")

    print(f"Threshold: {float(reference_row['Threshold']):.8f}")

    print(

        "Confusion matrix: "

        f"TP={int(reference_row['TP'])}, "

        f"FN={int(reference_row['FN'])}, "

        f"FP={int(reference_row['FP'])}, "

        f"TN={int(reference_row['TN'])}"

    )

    print(f"Sensitivity: {float(reference_row['Recall']):.4f}")

    print(f"Specificity: {float(reference_row['Specificity']):.4f}")

    print(f"PPV: {float(reference_row['Precision']):.4f}")

    print(f"Flagged encounters: {reference_flagged_percent:.2f}%")

    print(f"Alerts per 1,000: {reference_alerts:.2f}")

    print(

        "Deaths detected: "

        f"{int(reference_row['TP'])}/{n_events}"

    )



    print("\nCALIBRATION TABLE")

    print("-" * 100)

    print(

        calibration_values[

            [

                "model",

                "bin",

                "n",

                "events",

                "mean_predicted",

                "observed_rate",

                "CI_low",

                "CI_high",

            ]

        ]

        .round(6)

        .to_string(index=False)

    )



    print("\nSAVED OUTPUTS")

    print("-" * 100)

    for path in [

        png_path, pdf_path, svg_path, tiff_path, metrics_out, operating_out,

        calibration_out, dca_out, paired_out, summary_out, provenance_out,

        numerics_out,

    ]:

        if path.exists():

            print(path)



    return {

        "png": png_path,

        "pdf": pdf_path,

        "svg": svg_path,

        "tiff": tiff_path,

        "metrics": metrics_out,

        "operating": operating_out,

        "calibration": calibration_out,

        "decision_curve": dca_out,

        "summary": summary_out,

        "provenance": provenance_out,

        "manuscript_numerics": numerics_out,

    }





# =============================================================================

# COMMAND-LINE ENTRY POINT

# =============================================================================





def parse_args(argv=None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Create the final temporal-safe strict-OOF 3x2 performance and clinical-utility Figure 3."

    )

    parser.add_argument(

        "--run_dir",

        type=pathlib.Path,

        default=DEFAULT_RUN_DIR,

        help="Completed temporal-safe strict-OOF base-run directory.",

    )

    parser.add_argument(

        "--output_dir",

        type=pathlib.Path,

        default=None,

        help="Output directory. Default: <run_dir>/figures_revised_manuscript/figure3_performance_strict_oof",

    )

    parser.add_argument(

        "--calibration_bootstraps",

        type=int,

        default=CALIBRATION_BOOTSTRAPS,

        help="Stratified bootstrap replicates for the hybrid-blend calibration CI.",

    )

    parser.add_argument(

        "--no_show",

        action="store_true",

        help=(

            "Save the figure without displaying it inline. "

            "By default, plt.show() displays the figure in Jupyter."

        ),

    )

    # Jupyter/IPython injects arguments such as ``--f=<kernel.json>``.

    # parse_known_args preserves the documented CLI options while safely

    # ignoring notebook-only arguments.

    args, unknown = parser.parse_known_args(argv)

    if unknown:

        print(

            "Ignoring unrecognized Jupyter/IPython arguments: "

            + " ".join(map(str, unknown))

        )

    return args





def main() -> None:

    args = parse_args()

    output_dir = args.output_dir or (

        args.run_dir / "figures_revised_manuscript" / "figure3_performance_strict_oof"

    )

    build_figure(

        run_dir=args.run_dir,

        output_dir=output_dir,

        calibration_bootstraps=args.calibration_bootstraps,

        show_figure=not args.no_show,

    )





if __name__ == "__main__":

    main()