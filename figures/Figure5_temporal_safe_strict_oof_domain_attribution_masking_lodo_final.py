"""

FIGURE 5 — DOMAIN ATTRIBUTION, INPUT MASKING,

AND LEAVE-ONE-DOMAIN-OUT PERFORMANCE

================================================================



Final temporal-safe, strict-OOF run

-------------------------

/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/

darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653



This script reads only saved artifacts from the completed run. It does not

train, recalibrate, mask, or refit a model.



Analyses are intentionally kept distinct

----------------------------------------

1. Domain attribution:

   - Direct GradientSHAP for the Uniform DARN component.

   - Exact TreeSHAP for the XGBoost component.



2. Fitted-model input-domain masking:

   - Selected transformed features belonging to one domain were set to zero at

     inference while the already-fitted Uniform DARN models were retained.

   - This is a sensitivity analysis, not retraining.



3. True leave-one-domain-out retraining:

   - All raw predictors belonging to one domain were removed.

   - Preprocessing, variance filtering, feature selection, five-fold Uniform

     DARN fitting, XGBoost fitting, calibration, hybrid blending, and threshold

     selection were repeated using development data.

   - Paired stratified-bootstrap differences are evaluated on the same locked

     holdout encounters. Positive differences favor the complete model.



Main panels

-----------

A. Normalized domain attribution for Uniform DARN.

B. Normalized domain attribution for XGBoost.

C. Uniform DARN fitted-model input-domain masking sensitivity.

D. True drop-one-domain AUPRC changes for Uniform DARN and the hybrid blend.

E. True drop-one-domain AUROC changes for Uniform DARN and the hybrid blend.

F. Uniform DARN attribution versus AUPRC loss after true domain-removal

   retraining.



The drop-one panels use:

leave_one_domain_out_retraining/

    leave_one_domain_out_paired_bootstrap_differences.csv



The script is compatible with command-line execution and Jupyter `%run`.

"""



from __future__ import annotations



import argparse

import json

import pathlib

import sys

import warnings

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib import cm

from matplotlib.patches import Patch



warnings.filterwarnings("ignore")





# =============================================================================

# CURRENT-RUN SETTINGS

# =============================================================================



DEFAULT_RUN_DIR = pathlib.Path(

    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"

    "darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653"

)



DEFAULT_OUTPUT_SUBDIR = (

    "figures_revised_manuscript/"

    "figure5_domain_attribution_masking_lodo_strict_oof"

)



CMAP_NAME = "plasma"



DROP_ONE_MODEL_ORDER = [

    "Hybrid DARN-XGB Blend",

    "Uniform DARN",

]



DROP_ONE_MODEL_DISPLAY = {

    "Hybrid DARN-XGB Blend": "Hybrid blend",

    "Uniform DARN": "Uniform DARN",

}



DROP_ONE_MARKERS = {

    "Hybrid DARN-XGB Blend": "o",

    "Uniform DARN": "s",

}



DOMAIN_DISPLAY_MAP: Dict[str, str] = {

    "Demographics": "Demographics",

    "Diagnoses": "Diagnoses",

    "Durations": "Durations",

    "Interactions": "Interactions",

    "Emergency_Interactions": "Interactions",

    "Intraop_Vitals": "Intraoperative variables",

    "Medications": "Medications",

    "Preop_Labs": "Preoperative laboratories",

    "Procedures": "Procedures",

    "Other": "Other",

}



DOMAIN_CANONICAL_MAP: Dict[str, str] = {

    "Emergency_Interactions": "Interactions",

    "Emergency interactions": "Interactions",

    "Preoperative laboratories": "Preop_Labs",

    "Intraoperative vitals": "Intraop_Vitals",

    "Intraoperative variables": "Intraop_Vitals",

}



MAIN_BASENAME = "Figure5_Domain_Attribution_Masking_and_LODO_StrictOOF"





# =============================================================================

# HELPERS

# =============================================================================



def require_file(path: pathlib.Path) -> pathlib.Path:

    if not path.exists():

        raise FileNotFoundError(f"Required file not found:\n{path}")

    return path





def first_existing(

    candidates: Iterable[pathlib.Path],

    description: str,

) -> pathlib.Path:

    checked: List[pathlib.Path] = []



    for path in candidates:

        checked.append(path)

        if path.exists():

            return path



    checked_text = "\n".join(f"  - {path}" for path in checked)

    raise FileNotFoundError(

        f"Could not locate {description}. Checked:\n{checked_text}"

    )





def canonical_domain(domain: Any) -> str:

    raw = str(domain).strip()

    return DOMAIN_CANONICAL_MAP.get(raw, raw)





def display_domain(domain: Any) -> str:

    canonical = canonical_domain(domain)

    return DOMAIN_DISPLAY_MAP.get(

        canonical,

        canonical.replace("_", " "),

    )





def wrap_domain(label: str) -> str:

    replacements = {

        "Preoperative laboratories": "Preoperative\nlaboratories",

        "Intraoperative variables": "Intraoperative\nvariables",

    }

    return replacements.get(label, label)





def make_domain_colors(domains: Sequence[str]) -> Dict[str, Any]:

    cmap = cm.get_cmap(CMAP_NAME)

    positions = np.linspace(

        0.08,

        0.92,

        max(len(domains), 2),

    )

    return {

        domain: cmap(position)

        for domain, position in zip(domains, positions)

    }





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

        raise TypeError(f"Cannot serialize {type(value)}")



    with open(path, "w", encoding="utf-8") as handle:

        json.dump(data, handle, indent=2, default=convert)





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





def configure_style() -> None:
    """Use the same publication typography as Figures 2–4."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "figure.titlesize": 18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# FINAL STRICT-OOF FILE DISCOVERY
# =============================================================================


def discover_files(run_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Locate artifacts belonging to the final strict-OOF run only."""
    revised = run_dir / "figures_revised_manuscript"
    figure4_dir = revised / "figure4_darn_xgboost_AB_strict_oof"

    darn_feature_file = require_file(
        figure4_dir / "uniform_darn_gradientshap_feature_importance.csv"
    )
    xgb_feature_file = require_file(
        figure4_dir / "xgboost_treeshap_feature_importance.csv"
    )

    masking_file = first_existing(
        [
            run_dir / "darn_input_domain_masking.csv",
            run_dir / "darn_domain_ablation.csv",
            run_dir / "tables" / "darn_input_domain_masking.csv",
            run_dir / "tables" / "darn_domain_ablation.csv",
        ],
        "the strict-OOF fitted-model input-domain masking table",
    )

    lodo_directory = run_dir / "leave_one_domain_out_retraining"
    lodo_paired_file = require_file(
        lodo_directory / "leave_one_domain_out_paired_bootstrap_differences.csv"
    )
    lodo_metrics_file = require_file(
        lodo_directory / "leave_one_domain_out_retrained_test_metrics.csv"
    )

    completion_file = first_existing(
        [
            run_dir / "PARALLEL_RETRAINING_COMPLETE.json",
            lodo_directory / "PARALLEL_RETRAINING_COMPLETE.json",
        ],
        "the strict-OOF leave-one-domain-out completion marker",
    )

    with open(completion_file, "r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if not bool(completion.get("completed", False)):
        raise RuntimeError(
            f"{completion_file.name} does not report a completed retraining workflow."
        )

    return {
        "darn_feature_file": darn_feature_file,
        "xgb_feature_file": xgb_feature_file,
        "masking_file": masking_file,
        "lodo_paired_file": lodo_paired_file,
        "lodo_metrics_file": lodo_metrics_file,
        "completion_file": completion_file,
    }


# =============================================================================
# DOMAIN ATTRIBUTION
# =============================================================================


def identify_importance_column(

    table: pd.DataFrame,

    filename: str,

) -> str:

    candidates = [

        "mean_absolute_direct_attribution",

        "mean_absolute_attribution",

        "mean_abs_attribution",

        "mean_abs_shap",

        "mean_absolute_shap",

        "importance",

    ]



    for column in candidates:

        if column in table.columns:

            return column



    raise KeyError(

        f"{filename} does not contain a recognized attribution column. "

        f"Available columns: {list(table.columns)}"

    )





def aggregate_attribution_by_domain(

    path: pathlib.Path,

    component: str,

) -> pd.DataFrame:

    table = pd.read_csv(require_file(path))



    if "domain" not in table.columns:

        raise KeyError(

            f"{path.name} does not contain a 'domain' column."

        )



    importance_column = identify_importance_column(

        table,

        path.name,

    )



    table = table.copy()

    table["domain"] = table["domain"].map(

        canonical_domain

    )

    table[importance_column] = pd.to_numeric(

        table[importance_column],

        errors="coerce",

    ).fillna(0.0)



    domain = (

        table.groupby("domain", as_index=False)

        .agg(

            absolute_attribution=(

                importance_column,

                "sum",

            ),

            selected_features=(

                importance_column,

                "size",

            ),

        )

    )



    total = float(

        domain["absolute_attribution"].sum()

    )

    if total <= 0:

        raise ValueError(

            f"Total attribution is zero in {path.name}."

        )



    domain["attribution_percent"] = (

        domain["absolute_attribution"]

        / total

        * 100.0

    )

    domain["component"] = component

    domain["display_domain"] = domain["domain"].map(

        display_domain

    )



    return domain





# =============================================================================

# REVIEWER-v6 FITTED-MODEL INPUT-MASKING TABLE

# =============================================================================



def load_input_masking_table(

    path: pathlib.Path,

) -> pd.DataFrame:

    """

    Normalize the strict-OOF fitted-model input-domain masking schema.



    Expected columns:

        domain

        baseline_AUROC

        ablated_AUROC

        AUROC_drop

        baseline_AUPRC

        ablated_AUPRC

        AUPRC_drop

    """

    table = pd.read_csv(require_file(path))



    required = {

        "domain",

        "baseline_AUROC",

        "ablated_AUROC",

        "AUROC_drop",

        "baseline_AUPRC",

        "ablated_AUPRC",

        "AUPRC_drop",

    }



    missing = required - set(table.columns)

    if missing:

        raise KeyError(

            f"{path.name} is missing columns: {sorted(missing)}\n"

            f"Available columns: {list(table.columns)}"

        )



    table = table.copy()

    table["domain"] = table["domain"].map(

        canonical_domain

    )



    numeric_columns = [

        "baseline_AUROC",

        "ablated_AUROC",

        "AUROC_drop",

        "baseline_AUPRC",

        "ablated_AUPRC",

        "AUPRC_drop",

    ]



    for column in numeric_columns:

        table[column] = pd.to_numeric(

            table[column],

            errors="coerce",

        )



    table["display_domain"] = table["domain"].map(

        display_domain

    )



    # Positive change = discrimination declined after masking.

    table["AUPRC_change_after_masking"] = table["AUPRC_drop"]

    table["AUROC_change_after_masking"] = table["AUROC_drop"]



    return table.sort_values(

        "AUPRC_change_after_masking",

        ascending=False,

    ).reset_index(drop=True)





# =============================================================================

# TRUE LEAVE-ONE-DOMAIN-OUT RETRAINING

# =============================================================================



def load_drop_one_table(path: pathlib.Path) -> pd.DataFrame:

    """

    Normalize the paired-bootstrap leave-one-domain-out retraining results.



    Positive differences always favor the complete model:

      AUROC/AUPRC:

          complete model - domain-removed retrained model

      Brier/absolute O:E error:

          domain-removed retrained model - complete model

    """

    table = pd.read_csv(require_file(path))



    domain_column = None

    for candidate in ["removed_domain", "removed_feature_set"]:

        if candidate in table.columns:

            domain_column = candidate

            break



    if domain_column is None:

        raise KeyError(

            f"{path.name} has no removed-domain column. "

            f"Available columns: {list(table.columns)}"

        )



    required = {

        "model",

        "metric",

        "full_estimate",

        "retrained_estimate",

        "difference_favoring_complete_model",

        "CI_low",

        "CI_high",

    }

    missing = required - set(table.columns)

    if missing:

        raise KeyError(

            f"{path.name} is missing columns: {sorted(missing)}\n"

            f"Available columns: {list(table.columns)}"

        )



    table = table.copy()

    table["domain"] = table[domain_column].map(canonical_domain)

    table["display_domain"] = table["domain"].map(display_domain)



    numeric_columns = [

        "full_estimate",

        "retrained_estimate",

        "difference_favoring_complete_model",

        "CI_low",

        "CI_high",

        "bootstrap_p_two_sided",

    ]

    for column in numeric_columns:

        if column in table.columns:

            table[column] = pd.to_numeric(

                table[column],

                errors="coerce",

            )



    table["significant_95"] = (

        (table["CI_low"] > 0)

        | (table["CI_high"] < 0)

    )



    table["performance_change"] = table[

        "difference_favoring_complete_model"

    ]


    return table.sort_values(

        [

            "metric",

            "model",

            "performance_change",

        ],

        ascending=[True, True, False],

    ).reset_index(drop=True)





def load_drop_one_metrics(path: pathlib.Path) -> pd.DataFrame:

    """Load the direct metrics table for numerical provenance."""

    table = pd.read_csv(require_file(path))



    domain_column = (

        "removed_domain"

        if "removed_domain" in table.columns

        else "variant"

        if "variant" in table.columns

        else None

    )



    if domain_column is not None:

        table = table.copy()

        table["domain"] = table[domain_column].map(canonical_domain)

        table["display_domain"] = table["domain"].map(display_domain)



    return table





def drop_one_domain_order(

    table: pd.DataFrame,

    reference_model: str = "Hybrid DARN-XGB Blend",

) -> List[str]:

    """Order domains by AUPRC change for the reference retrained model."""

    available_models = table["model"].astype(str).unique().tolist()

    model = (

        reference_model

        if reference_model in available_models

        else "Uniform DARN"

        if "Uniform DARN" in available_models

        else available_models[0]

    )



    order = (

        table.loc[

            table["model"].eq(model)

            & table["metric"].eq("AUPRC")

        ]

        .sort_values(

            "performance_change",

            ascending=False,

        )["domain"]

        .drop_duplicates()

        .tolist()

    )



    if not order:

        order = sorted(table["domain"].dropna().unique().tolist())



    return order





def create_retraining_agreement_table(

    darn_domain: pd.DataFrame,

    drop_one: pd.DataFrame,

) -> tuple[pd.DataFrame, float]:

    """Compare DARN domain attribution with true DARN retraining loss."""

    retraining = (

        drop_one.loc[

            drop_one["model"].eq("Uniform DARN")

            & drop_one["metric"].eq("AUPRC"),

            [

                "domain",

                "full_estimate",

                "retrained_estimate",

                "performance_change",

                "CI_low",

                "CI_high",

                "bootstrap_p_two_sided",

                "significant_95",

            ],

        ]

        .rename(

            columns={

                "performance_change": "AUPRC_drop_after_retraining",

                "CI_low": "AUPRC_drop_CI_low",

                "CI_high": "AUPRC_drop_CI_high",

            }

        )

    )



    agreement = (

        darn_domain[

            [

                "domain",

                "display_domain",

                "attribution_percent",

                "selected_features",

            ]

        ]

        .merge(

            retraining,

            on="domain",

            how="inner",

        )

        .sort_values(

            "attribution_percent",

            ascending=False,

        )

        .reset_index(drop=True)

    )



    if len(agreement) >= 3:

        x_rank = (

            agreement["attribution_percent"]

            .rank(method="average")

            .to_numpy(dtype=float)

        )

        y_rank = (

            agreement["AUPRC_drop_after_retraining"]

            .rank(method="average")

            .to_numpy(dtype=float)

        )

        spearman_rho = float(np.corrcoef(x_rank, y_rank)[0, 1])

    else:

        spearman_rho = float("nan")



    return agreement, spearman_rho





def create_masking_retraining_comparison(

    masking: pd.DataFrame,

    drop_one: pd.DataFrame,

) -> pd.DataFrame:

    """Save a direct comparison without conflating masking and retraining."""

    retraining = (

        drop_one.loc[

            drop_one["model"].eq("Uniform DARN")

            & drop_one["metric"].isin(["AUPRC", "AUROC"]),

            [

                "domain",

                "metric",

                "performance_change",

                "CI_low",

                "CI_high",

            ],

        ]

        .pivot(

            index="domain",

            columns="metric",

            values=[

                "performance_change",

                "CI_low",

                "CI_high",

            ],

        )

    )



    retraining.columns = [

        f"retraining_{metric}_{statistic}"

        for statistic, metric in retraining.columns

    ]

    retraining = retraining.reset_index()



    comparison = masking[

        [

            "domain",

            "display_domain",

            "AUPRC_change_after_masking",

            "AUROC_change_after_masking",

        ]

    ].merge(

        retraining,

        on="domain",

        how="outer",

    )



    return comparison.sort_values(

        "AUPRC_change_after_masking",

        ascending=False,

        na_position="last",

    ).reset_index(drop=True)





# =============================================================================

# PANEL DRAWING

# =============================================================================



def draw_attribution_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    domain_colors: Mapping[str, Any],

    *,

    panel_letter: str,

    title: str,

) -> None:

    panel = table.sort_values(

        "attribution_percent",

        ascending=True,

    ).reset_index(drop=True)



    bars = ax.barh(

        panel["display_domain"],

        panel["attribution_percent"],

        color=[

            domain_colors[domain]

            for domain in panel["domain"]

        ],

        edgecolor="none",

    )



    maximum = max(

        float(panel["attribution_percent"].max()),

        1e-9,

    )

    ax.set_xlim(0, maximum * 1.20)



    for bar, value in zip(

        bars,

        panel["attribution_percent"],

    ):

        ax.text(

            float(value) + maximum * 0.014,

            bar.get_y() + bar.get_height() / 2,

            f"{float(value):.1f}%",

            va="center",

            ha="left",

            fontsize=10.5,

        )



    ax.set_xlabel(

        "Share of total absolute attribution (%)"

    )

    ax.set_title(

        f"{panel_letter}. {title}",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        axis="x",

        linestyle="--",

        alpha=0.23,

    )





def draw_masking_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    domain_colors: Mapping[str, Any],

    *,

    panel_letter: str,

) -> None:

    panel = table.sort_values(

        "AUPRC_change_after_masking",

        ascending=True,

    ).reset_index(drop=True)



    y_positions = np.arange(len(panel))

    height = 0.34



    colors = [

        domain_colors[domain]

        for domain in panel["domain"]

    ]



    auprc_bars = ax.barh(

        y_positions - height / 2,

        panel["AUPRC_change_after_masking"],

        height=height,

        color=colors,

        alpha=0.95,

        edgecolor="none",

        label="AUPRC change",

    )



    auroc_bars = ax.barh(

        y_positions + height / 2,

        panel["AUROC_change_after_masking"],

        height=height,

        color=colors,

        alpha=0.42,

        edgecolor="none",

        label="AUROC change",

    )



    ax.axvline(

        0,

        color="dimgray",

        linestyle="--",

        linewidth=1.0,

    )



    ax.set_yticks(y_positions)

    ax.set_yticklabels(panel["display_domain"])

    ax.set_xlabel(

        "Baseline discrimination − discrimination after masking"

    )

    ax.set_title(

        f"{panel_letter}. Uniform DARN input-domain masking",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        axis="x",

        linestyle="--",

        alpha=0.23,

    )



    maximum = max(

        float(

            np.nanmax(

                np.abs(

                    panel[

                        [

                            "AUPRC_change_after_masking",

                            "AUROC_change_after_masking",

                        ]

                    ].to_numpy(dtype=float)

                )

            )

        ),

        1e-6,

    )



    lower = min(

        float(

            np.nanmin(

                panel[

                    [

                        "AUPRC_change_after_masking",

                        "AUROC_change_after_masking",

                    ]

                ].to_numpy(dtype=float)

            )

        ),

        0.0,

    )

    upper = max(

        float(

            np.nanmax(

                panel[

                    [

                        "AUPRC_change_after_masking",

                        "AUROC_change_after_masking",

                    ]

                ].to_numpy(dtype=float)

            )

        ),

        0.0,

    )

    padding = maximum * 0.18

    ax.set_xlim(lower - padding, upper + padding)



    for bars in [auprc_bars, auroc_bars]:

        for bar in bars:

            value = float(bar.get_width())

            offset = maximum * 0.022

            ax.text(

                value + (

                    offset

                    if value >= 0

                    else -offset

                ),

                bar.get_y() + bar.get_height() / 2,

                f"{value:+.3f}",

                va="center",

                ha=(

                    "left"

                    if value >= 0

                    else "right"

                ),

                fontsize=10.0,

            )



    legend_handles = [

        Patch(

            facecolor="#666666",

            alpha=0.95,

            edgecolor="none",

            label="AUPRC change",

        ),

        Patch(

            facecolor="#666666",

            alpha=0.42,

            edgecolor="none",

            label="AUROC change",

        ),

    ]



    ax.legend(

        handles=legend_handles,

        frameon=False,

        fontsize=10.5,

        loc="lower right",

    )







def draw_drop_one_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    domain_order: Sequence[str],

    domain_colors: Mapping[str, Any],

    *,

    metric: str,

    panel_letter: str,

) -> None:

    """

    Draw true leave-one-domain-out performance changes with paired-bootstrap

    95% confidence intervals.

    """

    panel = table.loc[

        table["metric"].eq(metric)

        & table["model"].isin(DROP_ONE_MODEL_ORDER)

    ].copy()



    if panel.empty:

        raise ValueError(

            f"No drop-one-domain rows were found for metric {metric}."

        )



    available_models = [

        model

        for model in DROP_ONE_MODEL_ORDER

        if model in panel["model"].unique()

    ]



    y_base = np.arange(len(domain_order), dtype=float)

    offsets = (

        np.linspace(-0.16, 0.16, len(available_models))

        if len(available_models) > 1

        else np.array([0.0])

    )



    legend_handles = []



    for model_index, model_name in enumerate(available_models):

        model_table = (

            panel.loc[panel["model"].eq(model_name)]

            .drop_duplicates("domain")

            .set_index("domain")

            .reindex(domain_order)

            .reset_index()

        )



        valid = model_table["performance_change"].notna().to_numpy()

        y_values = y_base + offsets[model_index]

        x_values = model_table["performance_change"].to_numpy(dtype=float)



        low = model_table["CI_low"].to_numpy(dtype=float)

        high = model_table["CI_high"].to_numpy(dtype=float)

        lower_error = np.maximum(x_values - low, 0.0)

        upper_error = np.maximum(high - x_values, 0.0)



        for row_number, domain in enumerate(domain_order):

            if not valid[row_number]:

                continue



            ax.errorbar(

                x_values[row_number],

                y_values[row_number],

                xerr=np.array(

                    [

                        [lower_error[row_number]],

                        [upper_error[row_number]],

                    ]

                ),

                fmt=DROP_ONE_MARKERS.get(model_name, "o"),

                markersize=7.0,

                capsize=2.8,

                elinewidth=1.35,

                linewidth=0,

                color=domain_colors[domain],

                markerfacecolor=domain_colors[domain],

                markeredgecolor="white",

                markeredgewidth=0.75,

                zorder=3,

            )



        legend_handles.append(

            mpl.lines.Line2D(

                [0],

                [0],

                marker=DROP_ONE_MARKERS.get(model_name, "o"),

                color="dimgray",

                markerfacecolor="dimgray",

                markeredgecolor="white",

                linewidth=0,

                markersize=7,

                label=DROP_ONE_MODEL_DISPLAY.get(model_name, model_name),

            )

        )



    all_ci = panel[["CI_low", "CI_high"]].to_numpy(dtype=float)

    finite_ci = all_ci[np.isfinite(all_ci)]



    if finite_ci.size:

        minimum = min(float(finite_ci.min()), 0.0)

        maximum = max(float(finite_ci.max()), 0.0)
    else:

        values = panel["performance_change"].to_numpy(dtype=float)

        minimum = min(float(np.nanmin(values)), 0.0)

        maximum = max(float(np.nanmax(values)), 0.0)



    span = max(maximum - minimum, 0.002)

    padding = 0.12 * span



    ax.axvline(

        0,

        color="dimgray",

        linestyle="--",

        linewidth=1.0,

        zorder=1,

    )

    ax.set_xlim(minimum - padding, maximum + padding)

    ax.set_yticks(y_base)

    ax.set_yticklabels(

        [

            display_domain(domain)

            for domain in domain_order

        ]

    )

    ax.invert_yaxis()



    metric_label = (

        "AUPRC"

        if metric == "AUPRC"

        else "AUROC"

    )

    ax.set_xlabel(

        f"Complete model − domain-removed model ({metric_label})"

    )

    ax.set_title(

        f"{panel_letter}. Leave-one-domain-out {metric_label} change",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        axis="x",

        linestyle="--",

        alpha=0.23,

    )

    ax.legend(

        handles=legend_handles,

        frameon=False,

        fontsize=10.5,

        loc="best",

    )





def draw_retraining_agreement_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    domain_colors: Mapping[str, Any],

    *,

    spearman_rho: float,

    panel_letter: str,

) -> None:

    """Attribution versus true Uniform DARN AUPRC loss after retraining."""

    feature_counts = (

        table["selected_features"]

        .fillna(1)

        .clip(lower=1)

        .to_numpy(dtype=float)

    )

    bubble_sizes = (

        120

        + 700

        * feature_counts

        / max(float(feature_counts.max()), 1.0)

    )



    x = table["attribution_percent"].to_numpy(dtype=float)

    y = table["AUPRC_drop_after_retraining"].to_numpy(dtype=float)

    low = table["AUPRC_drop_CI_low"].to_numpy(dtype=float)

    high = table["AUPRC_drop_CI_high"].to_numpy(dtype=float)



    yerr = np.vstack(

        [

            np.maximum(y - low, 0.0),

            np.maximum(high - y, 0.0),

        ]

    )



    ax.errorbar(

        x,

        y,

        yerr=yerr,

        fmt="none",

        ecolor="gray",

        elinewidth=1.0,

        capsize=2.2,

        alpha=0.65,

        zorder=1,

    )

    ax.scatter(

        x,

        y,

        s=bubble_sizes,

        c=[

            domain_colors[domain]

            for domain in table["domain"]

        ],

        edgecolors="black",

        linewidths=0.55,

        alpha=0.87,

        zorder=2,

    )



    label_offsets = {
        "Demographics": (-10, 16),
        "Diagnoses": (8, -18),
        "Durations": (-10, -14),
        "Interactions": (8, -14),
        "Intraop_Vitals": (8, 10),
        "Medications": (-12, 16),
        "Preop_Labs": (8, -16),
        "Procedures": (8, 11),
    }



    for index, row in table.reset_index(drop=True).iterrows():

        dx, dy = label_offsets.get(
            canonical_domain(row["domain"]),
            (8, 8),
        )

        ax.annotate(

            row["display_domain"],

            (

                row["attribution_percent"],

                row["AUPRC_drop_after_retraining"],

            ),

            xytext=(dx, dy),

            textcoords="offset points",

            fontsize=9.6,

            ha="left" if dx >= 0 else "right",

            va="center",

            arrowprops={

                "arrowstyle": "-",

                "lw": 0.5,

                "alpha": 0.45,

            },

        )



    ax.axhline(

        0,

        color="dimgray",

        linestyle="--",

        linewidth=1.0,

    )

    ax.set_xlabel(

        "Uniform DARN normalized domain attribution (%)"

    )

    ax.set_ylabel(

        "Complete − domain-removed AUPRC"

    )

    ax.set_title(

        f"{panel_letter}. Attribution–retraining agreement",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        linestyle="--",

        alpha=0.22,

    )
    ax.margins(x=0.08, y=0.12)



    rho_text = (

        f"Spearman ρ = {spearman_rho:.2f}"

        if np.isfinite(spearman_rho)

        else "Spearman ρ unavailable"

    )

    ax.text(

        0.03,

        0.96,

        rho_text,

        transform=ax.transAxes,

        ha="left",

        va="top",

        fontsize=10.5,

        color="dimgray",

    )





# =============================================================================

# SUPPLEMENTARY AGREEMENT

# =============================================================================



def create_agreement_table(

    darn_domain: pd.DataFrame,

    masking: pd.DataFrame,

) -> pd.DataFrame:

    return (

        darn_domain[

            [

                "domain",

                "display_domain",

                "attribution_percent",

                "selected_features",

            ]

        ]

        .merge(

            masking[

                [

                    "domain",

                    "AUPRC_change_after_masking",

                    "AUROC_change_after_masking",

                ]

            ],

            on="domain",

            how="inner",

        )

        .sort_values(

            "attribution_percent",

            ascending=False,

        )

        .reset_index(drop=True)

    )





def draw_agreement_panel(

    ax: plt.Axes,

    table: pd.DataFrame,

    domain_colors: Mapping[str, Any],

    *,

    panel_letter: str,

) -> None:

    """Compare DARN domain attribution with AUPRC change after masking."""

    feature_counts = (

        table["selected_features"]

        .fillna(1)

        .clip(lower=1)

        .to_numpy(dtype=float)

    )

    bubble_sizes = (

        120

        + 700

        * feature_counts

        / max(float(feature_counts.max()), 1.0)

    )



    ax.scatter(

        table["attribution_percent"],

        table["AUPRC_change_after_masking"],

        s=bubble_sizes,

        c=[

            domain_colors[domain]

            for domain in table["domain"]

        ],

        edgecolors="black",

        linewidths=0.55,

        alpha=0.85,

    )



    offsets = [

        (7, 8),

        (7, -9),

        (-7, 8),

        (-7, -9),

        (9, 0),

        (-9, 0),

        (6, 12),

        (-6, -12),

    ]



    for index, row in table.reset_index(drop=True).iterrows():

        dx, dy = offsets[index % len(offsets)]

        ax.annotate(

            row["display_domain"],

            (

                row["attribution_percent"],

                row["AUPRC_change_after_masking"],

            ),

            xytext=(dx, dy),

            textcoords="offset points",

            fontsize=10.0,

            ha="left" if dx >= 0 else "right",

            va="center",

            arrowprops={

                "arrowstyle": "-",

                "lw": 0.55,

                "alpha": 0.50,

            },

        )



    ax.axhline(

        0,

        color="dimgray",

        linestyle="--",

        linewidth=1.0,

    )

    ax.set_xlabel(

        "Uniform DARN domain attribution (%)"

    )

    ax.set_ylabel(

        "AUPRC change after masking"

    )

    ax.set_title(

        f"{panel_letter}. Attribution–masking agreement",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        linestyle="--",

        alpha=0.22,

    )

    ax.text(

        0.98,

        0.04,

        (

            "Positive y: masking reduced AUPRC\n"

            "Negative y: masking improved AUPRC\n"

            "Bubble size: selected feature count"

        ),

        transform=ax.transAxes,

        ha="right",

        va="bottom",

        fontsize=9.8,

        color="dimgray",

        bbox={

            "boxstyle": "round",

            "facecolor": "white",

            "alpha": 0.75,

            "edgecolor": "lightgray",

        },

    )





def plot_agreement(

    table: pd.DataFrame,

    domain_colors: Mapping[str, Any],

    output_path: pathlib.Path,

) -> None:

    fig, ax = plt.subplots(

        figsize=(9.2, 7.2)

    )



    feature_counts = (

        table["selected_features"]

        .fillna(1)

        .clip(lower=1)

        .to_numpy(dtype=float)

    )

    bubble_sizes = (

        130

        + 800

        * feature_counts

        / max(float(feature_counts.max()), 1.0)

    )



    ax.scatter(

        table["attribution_percent"],

        table["AUPRC_change_after_masking"],

        s=bubble_sizes,

        c=[

            domain_colors[domain]

            for domain in table["domain"]

        ],

        edgecolors="black",

        linewidths=0.6,

        alpha=0.85,

    )



    offsets = [

        (8, 8),

        (8, -10),

        (-8, 8),

        (-8, -10),

        (10, 0),

        (-10, 0),

    ]



    for index, row in table.iterrows():

        dx, dy = offsets[index % len(offsets)]

        ax.annotate(

            row["display_domain"],

            (

                row["attribution_percent"],

                row["AUPRC_change_after_masking"],

            ),

            xytext=(dx, dy),

            textcoords="offset points",

            fontsize=10.5,

            ha=(

                "left"

                if dx >= 0

                else "right"

            ),

            va="center",

            arrowprops={

                "arrowstyle": "-",

                "lw": 0.6,

                "alpha": 0.55,

            },

        )



    ax.axhline(

        0,

        color="dimgray",

        linestyle="--",

        linewidth=1.0,

    )

    ax.set_xlabel(

        "Uniform DARN normalized domain attribution (%)"

    )

    ax.set_ylabel(

        "AUPRC change after input-domain masking"

    )

    ax.set_title(

        "Supplementary: attribution–masking agreement",

        loc="left",

        fontweight="bold",

    )

    ax.grid(

        linestyle="--",

        alpha=0.22,

    )



    fig.tight_layout()

    save_figure_all_formats(

        fig,

        output_path,

    )

    plt.show()

    plt.close(fig)





# =============================================================================

# MAIN WORKFLOW

# =============================================================================



def create_figure(

    run_dir: pathlib.Path,

    output_dir: pathlib.Path,

) -> None:

    configure_style()

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    files = discover_files(run_dir)



    print("\n" + "=" * 100)

    print(

        "FINAL TEMPORAL-SAFE STRICT-OOF DOMAIN ATTRIBUTION, INPUT MASKING, "

        "AND TRUE DROP-ONE RETRAINING"

    )

    print("=" * 100)

    print("Run directory        :", run_dir)

    print("Uniform DARN attribution:", files["darn_feature_file"])

    print("XGBoost attribution     :", files["xgb_feature_file"])

    print("Input masking           :", files["masking_file"])

    print("Drop-one paired bootstrap:", files["lodo_paired_file"])

    print("Drop-one metrics         :", files["lodo_metrics_file"])



    darn_domain = aggregate_attribution_by_domain(

        files["darn_feature_file"],

        "Uniform DARN",

    )

    xgb_domain = aggregate_attribution_by_domain(

        files["xgb_feature_file"],

        "XGBoost",

    )

    masking = load_input_masking_table(

        files["masking_file"]

    )

    drop_one = load_drop_one_table(

        files["lodo_paired_file"]

    )

    drop_one_metrics = load_drop_one_metrics(

        files["lodo_metrics_file"]

    )



    available_drop_one_models = [

        model

        for model in DROP_ONE_MODEL_ORDER

        if model in drop_one["model"].unique()

    ]

    if not available_drop_one_models:

        raise ValueError(

            "The paired-bootstrap drop-one file contains neither "

            "Hybrid DARN-XGB Blend nor Uniform DARN."

        )



    all_domains = sorted(

        set(darn_domain["domain"])

        .union(set(xgb_domain["domain"]))

        .union(set(masking["domain"]))

        .union(set(drop_one["domain"]))

    )



    average_attribution = (

        pd.concat(

            [

                darn_domain[

                    ["domain", "attribution_percent"]

                ],

                xgb_domain[

                    ["domain", "attribution_percent"]

                ],

            ],

            ignore_index=True,

        )

        .groupby("domain")["attribution_percent"]

        .mean()
        .sort_values(ascending=False)

    )



    color_order = (

        average_attribution.index.tolist()

        + [

            domain

            for domain in all_domains

            if domain not in average_attribution.index

        ]

    )

    domain_colors = make_domain_colors(color_order)



    performance_domain_order = drop_one_domain_order(

        drop_one,

        reference_model="Hybrid DARN-XGB Blend",

    )



    retraining_agreement, retraining_rho = (

        create_retraining_agreement_table(

            darn_domain,

            drop_one,

        )

    )



    masking_agreement = create_agreement_table(

        darn_domain,

        masking,

    )



    masking_retraining = create_masking_retraining_comparison(

        masking,

        drop_one,

    )



    # ------------------------------------------------------------------

    # Save all figure-specific source tables.

    # ------------------------------------------------------------------

    darn_domain.to_csv(

        output_dir

        / "Figure5A_Uniform_DARN_domain_attribution.csv",

        index=False,

    )

    xgb_domain.to_csv(

        output_dir

        / "Figure5B_XGBoost_domain_attribution.csv",

        index=False,

    )

    masking.to_csv(

        output_dir

        / "Figure5C_DARN_input_domain_masking.csv",

        index=False,

    )

    drop_one.to_csv(

        output_dir

        / "Figure5DE_true_drop_one_domain_paired_bootstrap.csv",

        index=False,

    )

    drop_one_metrics.to_csv(

        output_dir

        / "Figure5DE_true_drop_one_domain_test_metrics.csv",

        index=False,

    )

    retraining_agreement.to_csv(

        output_dir

        / "Figure5F_DARN_attribution_retraining_agreement.csv",

        index=False,

    )

    masking_agreement.to_csv(

        output_dir

        / "Supplementary_DARN_attribution_masking_agreement.csv",

        index=False,

    )

    masking_retraining.to_csv(

        output_dir

        / "Supplementary_DARN_masking_vs_retraining.csv",

        index=False,

    )



    # ------------------------------------------------------------------

    # Main 3 x 2 figure.

    # ------------------------------------------------------------------

    fig = plt.figure(

        figsize=(18.5, 18.0)

    )

    grid = fig.add_gridspec(

        3,

        2,

        width_ratios=[1.0, 1.08],

        height_ratios=[1.0, 1.0, 1.0],

        left=0.07,

        right=0.985,

        top=0.945,

        bottom=0.065,

        hspace=0.38,

        wspace=0.32,

    )



    ax_a = fig.add_subplot(grid[0, 0])

    ax_b = fig.add_subplot(grid[0, 1])

    ax_c = fig.add_subplot(grid[1, 0])

    ax_d = fig.add_subplot(grid[1, 1])

    ax_e = fig.add_subplot(grid[2, 0])

    ax_f = fig.add_subplot(grid[2, 1])



    draw_attribution_panel(

        ax_a,

        darn_domain,

        domain_colors,

        panel_letter="A",

        title="Uniform DARN domain attribution",

    )



    draw_attribution_panel(

        ax_b,

        xgb_domain,

        domain_colors,

        panel_letter="B",

        title="XGBoost domain attribution",

    )



    draw_masking_panel(

        ax_c,

        masking,

        domain_colors,

        panel_letter="C",

    )



    draw_drop_one_panel(

        ax_d,

        drop_one,

        performance_domain_order,

        domain_colors,

        metric="AUPRC",

        panel_letter="D",

    )



    draw_drop_one_panel(

        ax_e,

        drop_one,

        performance_domain_order,

        domain_colors,

        metric="AUROC",

        panel_letter="E",

    )



    draw_retraining_agreement_panel(

        ax_f,

        retraining_agreement,

        domain_colors,

        spearman_rho=retraining_rho,

        panel_letter="F",

    )



    fig.suptitle(

        (

            "Clinical-domain attribution, input masking, "

            "and leave-one-domain-out retraining"

        ),

        fontsize=18,

        fontweight="bold",

        y=0.982,

    )

    # Methodological distinctions are described in the figure caption.

    main_path = (

        output_dir

        / f"{MAIN_BASENAME}.png"

    )

    save_figure_all_formats(

        fig,

        main_path,

    )

    plt.show()

    plt.close(fig)



    # ------------------------------------------------------------------

    # Individual retraining panels for flexible manuscript assembly.

    # ------------------------------------------------------------------

    individual_specs = [

        (

            "D",

            "AUPRC",

            output_dir

            / "Figure5D_True_Drop_One_Domain_AUPRC.png",

        ),

        (

            "E",

            "AUROC",

            output_dir

            / "Figure5E_True_Drop_One_Domain_AUROC.png",

        ),

    ]



    for panel_letter, metric, path in individual_specs:

        figure, axis = plt.subplots(figsize=(10.2, 7.4))

        draw_drop_one_panel(

            axis,

            drop_one,

            performance_domain_order,

            domain_colors,

            metric=metric,

            panel_letter=panel_letter,

        )

        figure.tight_layout()

        save_figure_all_formats(

            figure,

            path,

        )

        plt.show()

        plt.close(figure)



    figure_f, axis_f = plt.subplots(figsize=(9.4, 7.4))

    draw_retraining_agreement_panel(

        axis_f,

        retraining_agreement,

        domain_colors,

        spearman_rho=retraining_rho,

        panel_letter="F",

    )

    figure_f.tight_layout()

    save_figure_all_formats(

        figure_f,

        output_dir

        / "Figure5F_DARN_Attribution_Retraining_Agreement.png",

    )

    plt.show()

    plt.close(figure_f)



    # Retain the old masking-agreement analysis only as a supplementary

    # sensitivity figure. It is not used as evidence of retraining.

    plot_agreement(

        masking_agreement,

        domain_colors,

        output_dir

        / "Supplementary_DARN_Attribution_Masking_Agreement.png",

    )



    summary = {

        "analysis_version": "temporal-safe-strict-oof",

        "run_id": run_dir.name,

        "run_directory": str(run_dir),

        "output_directory": str(output_dir),

        "uniform_darn_attribution_file": str(

            files["darn_feature_file"]

        ),

        "xgboost_attribution_file": str(

            files["xgb_feature_file"]

        ),

        "input_masking_file": str(

            files["masking_file"]

        ),

        "drop_one_paired_bootstrap_file": str(

            files["lodo_paired_file"]

        ),

        "drop_one_metrics_file": str(

            files["lodo_metrics_file"]

        ),

        "input_masking_method": (

            "Selected transformed features in one clinical domain were "

            "set to zero on the held-out test set while already-fitted "

            "fold-specific Uniform DARN models were retained."

        ),

        "input_masking_retraining_performed": False,

        "drop_one_method": (

            "One raw clinical domain was removed at a time. Preprocessing, "

            "variance filtering, supervised feature selection, five-fold "

            "Uniform DARN fitting, XGBoost fitting, calibration, hybrid "

            "blending, and threshold selection were repeated using "

            "development data. Performance differences were evaluated on "

            "the same held-out test cohort using paired stratified "

            "bootstrap resampling."

        ),

        "drop_one_retraining_performed": True,

        "drop_one_models": available_drop_one_models,

        "positive_drop_one_difference_definition": (

            "Positive AUROC/AUPRC differences equal complete model minus "

            "domain-removed retrained model and therefore favor the "

            "complete model."

        ),

        "uniform_darn_attribution_retraining_spearman_rho": (

            float(retraining_rho)

        ),

        "main_panels": {

            "A": "Normalized Uniform DARN GradientSHAP by domain",

            "B": "Normalized XGBoost TreeSHAP by domain",

            "C": "Uniform DARN fitted-model input-domain masking",

            "D": (

                "Leave-one-domain-out AUPRC differences with paired "

                "bootstrap 95% confidence intervals"

            ),

            "E": (

                "Leave-one-domain-out AUROC differences with paired "

                "bootstrap 95% confidence intervals"

            ),

            "F": (

                "Uniform DARN attribution versus AUPRC loss after true "

                "domain-removal retraining"

            ),

        },

        "main_figure": str(main_path),

        "interpretation_note": (

            "Attribution, input masking, and true retraining answer "

            "different questions. Attribution describes use by a fitted "

            "component; masking measures sensitivity of that fitted DARN "

            "to replacement by the transformed reference value; true "

            "drop-one retraining estimates performance after rebuilding "

            "the modeling pipeline without an entire raw domain."

        ),

    }



    save_json(

        summary,

        output_dir

        / "Figure5_temporal_safe_strict_oof_summary.json",

    )



    print("\n" + "=" * 100)

    print("FINAL TEMPORAL-SAFE STRICT-OOF FIGURE 5 COMPLETE")

    print("=" * 100)

    print("Main figure:", main_path)

    print("PDF:", main_path.with_suffix(".pdf"))

    print("SVG:", main_path.with_suffix(".svg"))

    print("Output directory:", output_dir)



    print("\nTRUE DROP-ONE AUPRC RESULTS")

    print("=" * 100)

    print(

        drop_one.loc[

            drop_one["metric"].eq("AUPRC")

            & drop_one["model"].isin(DROP_ONE_MODEL_ORDER),

            [

                "display_domain",

                "model",

                "full_estimate",

                "retrained_estimate",

                "performance_change",

                "CI_low",

                "CI_high",

                "bootstrap_p_two_sided",

            ],

        ]

        .sort_values(

            ["model", "performance_change"],

            ascending=[True, False],

        )

        .to_string(index=False)

    )



    print("\nUNIFORM DARN INPUT MASKING RESULTS")

    print("=" * 100)

    print(

        masking[

            [

                "display_domain",

                "baseline_AUPRC",

                "ablated_AUPRC",

                "AUPRC_change_after_masking",

                "baseline_AUROC",

                "ablated_AUROC",

                "AUROC_change_after_masking",

            ]

        ].to_string(index=False)

    )





# =============================================================================

# JUPYTER-SAFE ARGUMENT PARSING

# =============================================================================



def parse_args(

    argv: Optional[Sequence[str]] = None,

) -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        allow_abbrev=False,

        description=(

            "Generate the temporal-safe strict-OOF domain attribution, fitted-model "

            "input masking, and true leave-one-domain-out retraining figure."

        ),

    )



    parser.add_argument(

        "--run_dir",

        type=pathlib.Path,

        default=DEFAULT_RUN_DIR,

    )

    parser.add_argument(

        "--output_dir",

        type=pathlib.Path,

        default=None,

    )



    raw_arguments = list(

        sys.argv[1:]

        if argv is None

        else argv

    )



    filtered_arguments: List[str] = []

    skip_next = False



    for argument in raw_arguments:

        if skip_next:

            skip_next = False

            continue



        if argument in {"-f", "--f"}:

            skip_next = True

            continue



        if (

            argument.startswith("-f=")

            or argument.startswith("--f=")

        ):

            continue



        filtered_arguments.append(argument)



    args, unknown = parser.parse_known_args(

        filtered_arguments

    )



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



    create_figure(

        run_dir=args.run_dir,

        output_dir=output_dir,

    )





if __name__ == "__main__":

    main()