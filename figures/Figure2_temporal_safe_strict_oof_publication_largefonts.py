"""
FIGURE 2 — TEMPORAL-SAFE STRICT-OOF COHORT CHARACTERISTICS AND MORTALITY PATTERNS
===========================================================================

Completed temporal-safe strict-OOF run
-------------------------
/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/
darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653

Final pipeline
----------------
/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Code/New/
Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py

Main panels
-----------
A. Age distribution by 30-day outcome.
B. BMI distribution by 30-day outcome.
C. Sex distribution.
D. Aggregated ASA physical-status distribution.
E. Mortality by age group.
F. Mortality by ASA group.
G. Mortality by surgical urgency.
H. Mortality by major surgical department.
I. Mortality by sex.

Final-run reconstruction
--------------------------
The script:

1. Loads the exact temporal-safe strict-OOF `config.json`.
2. Applies the same final cohort exclusions used in the completed
   run, including exclusion of ASA physical status 6 before model splitting.
3. Reads `development_oof_predictions.csv` and
   `test_predictions_all_models.csv`.
4. Uses saved final-cohort row positions to verify the exact development and
   held-out test partitions.
5. Verifies that the development and test rows are unique, nonoverlapping,
   outcome-aligned, and exhaustive for the modeled cohort.
6. Builds the figure from the complete temporal-safe modeled cohort.

The temporal-safe strict-OOF analysis uses a development cohort with five-fold
cross-validation and a separate held-out test cohort. It does not use the
older train/validation/test structure.

Supplementary analysis
----------------------
The targeted eye/ear diagnosis cohort analysis is retained as a descriptive
supplementary figure and numerical table. It is distinct from the final
temporal-safe strict-OOF eye/ear feature-removal retraining analysis, which is saved under
`eye_ear_feature_retraining/`.

Run in Jupyter
--------------
%run Figure2_temporal_safe_strict_oof_cohort_characteristics.py
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
from matplotlib import cm

warnings.filterwarnings("ignore")


# =============================================================================
# DEFAULT CURRENT-RUN SETTINGS
# =============================================================================

DEFAULT_RUN_DIR = pathlib.Path(
    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"
    "darn_cv_baseline_runs/v6_parallel/run_20260719_123200"
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
    "figure2_cohort_characteristics_strict_oof"
)

FIGURE_BASENAME = "Figure2_Cohort_Characteristics_and_Mortality"
CMAP_NAME = "plasma"

TOP_DEPARTMENTS = 8
TOP_DIAGNOSES = 8
MIN_DEPARTMENT_N = 100

AGE_GROUP_ORDER = ["<60", "60–69", "≥70"]
ASA_GROUP_ORDER = ["I–II", "III", "IV–V"]
URGENCY_ORDER = ["Elective", "Emergency"]
SEX_GROUP_ORDER = ["Female", "Male"]

# INSPIRE encodes sex as a Boolean comparison with "M" in the official
# mortality example, so numeric/Boolean values are interpreted as:
# 0 / False = Female, 1 / True = Male.
SEX_NUMERIC_MAP: Dict[float, str] = {
    0.0: "Female",
    1.0: "Male",
}

STRICT_SEX_CODING = True


# =============================================================================
# FILE AND JSON HELPERS
# =============================================================================

def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired current-run file not found:\n{path}\n"
            "Confirm the completed temporal-safe strict-OOF run directory and paths."
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


def first_existing_file(
    candidates: Iterable[pathlib.Path],
    description: str,
) -> pathlib.Path:
    checked: List[pathlib.Path] = []

    for path in candidates:
        checked.append(path)
        if path.exists():
            return path

    checked_text = "\n".join(
        f"  - {path}"
        for path in checked
    )

    raise FileNotFoundError(
        f"Could not locate {description}. Checked:\n{checked_text}"
    )


def load_pipeline_module(module_path: pathlib.Path):
    require_file(module_path)

    spec = importlib.util.spec_from_file_location(
        "mortality_darn_v6_figure2_module",
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
# BASIC NUMERICAL HELPERS
# =============================================================================

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
        proportion
        + z**2 / (2.0 * n)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            proportion
            * (1.0 - proportion)
            / n
            + z**2
            / (4.0 * n**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def compact_label(
    text: str,
    max_words: int = 5,
) -> str:
    words = (
        str(text)
        .replace("_", " ")
        .split()
    )

    if len(words) <= max_words:
        return " ".join(words)

    return (
        " ".join(words[:max_words])
        + " …"
    )


def add_panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        -0.08,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )


# =============================================================================
# TEMPORAL-SAFE STRICT-OOF COHORT AND SPLIT RECONSTRUCTION
# =============================================================================

def prediction_position_column(
    prediction_table: pd.DataFrame,
) -> str:
    column = first_existing_column(
        prediction_table,
        [
            "final_cohort_row_position",
            "original_row_position",
        ],
    )

    if column is None:
        raise KeyError(
            "Prediction table lacks a final-cohort row-position column. "
            "Expected 'final_cohort_row_position' or "
            "'original_row_position'."
        )

    return column


def reconstruct_current_cohort(
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

    development_file = first_existing_file(
        [
            run_dir
            / "development_oof_predictions.csv",
            run_dir
            / "development_predictions.csv",
        ],
        "the current development prediction table",
    )

    test_file = require_file(
        run_dir
        / "test_predictions_all_models.csv"
    )

    development_predictions = pd.read_csv(
        development_file
    )
    test_predictions = pd.read_csv(
        test_file
    )

    development_position_column = (
        prediction_position_column(
            development_predictions
        )
    )
    test_position_column = (
        prediction_position_column(
            test_predictions
        )
    )

    development_positions = pd.to_numeric(
        development_predictions[
            development_position_column
        ],
        errors="raise",
    ).astype(int).to_numpy()

    test_positions = pd.to_numeric(
        test_predictions[
            test_position_column
        ],
        errors="raise",
    ).astype(int).to_numpy()

    if np.any(
        development_positions < 0
    ) or np.any(
        development_positions
        >= len(cohort)
    ):
        raise IndexError(
            "Development positions fall outside the temporal-safe cohort."
        )

    if np.any(
        test_positions < 0
    ) or np.any(
        test_positions
        >= len(cohort)
    ):
        raise IndexError(
            "Test positions fall outside the temporal-safe cohort."
        )

    if len(
        np.unique(
            development_positions
        )
    ) != len(
        development_positions
    ):
        raise RuntimeError(
            "Development row positions contain duplicates."
        )

    if len(
        np.unique(
            test_positions
        )
    ) != len(
        test_positions
    ):
        raise RuntimeError(
            "Test row positions contain duplicates."
        )

    overlap = np.intersect1d(
        development_positions,
        test_positions,
    )

    if len(overlap) > 0:
        raise RuntimeError(
            "Development and test positions overlap."
        )

    all_modeled_positions = np.concatenate(
        [
            development_positions,
            test_positions,
        ]
    )

    missing_positions = np.setdiff1d(
        np.arange(len(cohort)),
        all_modeled_positions,
    )

    if len(missing_positions) > 0:
        raise RuntimeError(
            f"{len(missing_positions):,} temporal-safe cohort rows are not "
            "represented in development or test predictions."
        )

    if len(all_modeled_positions) != len(cohort):
        raise RuntimeError(
            "Development and test sizes do not sum to the temporal-safe cohort."
        )

    target = clean_numeric(
        cohort[target_column]
    )

    if target.isna().any():
        raise ValueError(
            "The temporal-safe cohort contains missing outcome values."
        )

    target = target.astype(int)

    if not set(
        target.unique()
    ).issubset(
        {0, 1}
    ):
        raise ValueError(
            "Mortality target contains values other than 0 and 1."
        )

    development_outcomes = target.iloc[
        development_positions
    ].to_numpy()

    test_outcomes = target.iloc[
        test_positions
    ].to_numpy()

    if "y_true" in development_predictions.columns:
        if not np.array_equal(
            development_outcomes,
            development_predictions[
                "y_true"
            ].astype(int).to_numpy(),
        ):
            raise RuntimeError(
                "Development outcomes do not match saved OOF predictions."
            )

    if "y_true" in test_predictions.columns:
        if not np.array_equal(
            test_outcomes,
            test_predictions[
                "y_true"
            ].astype(int).to_numpy(),
        ):
            raise RuntimeError(
                "Test outcomes do not match saved predictions."
            )

    print("\n" + "=" * 100)
    print("TEMPORAL-SAFE STRICT-OOF CORRECTED MODELED COHORT")
    print("=" * 100)
    print(cohort_flow.to_string(index=False))
    print(f"Temporal-safe cohort: {len(cohort):,}")
    print(
        f"Development: {len(development_positions):,} "
        f"(deaths={int(development_outcomes.sum()):,})"
    )
    print(
        f"Test:     {len(test_positions):,} "
        f"(deaths={int(test_outcomes.sum()):,})"
    )
    print(
        f"Overall:     {len(cohort):,} "
        f"(deaths={int(target.sum()):,})"
    )

    expected_current_run = (
        run_dir.resolve()
        == DEFAULT_RUN_DIR.resolve()
    )
    if expected_current_run:
        expected_counts = {
            "cohort_n": 99834,
            "cohort_deaths": 235,
            "development_n": 79867,
            "development_deaths": 188,
            "test_n": 19967,
            "test_deaths": 47,
        }
        observed_counts = {
            "cohort_n": int(len(cohort)),
            "cohort_deaths": int(target.sum()),
            "development_n": int(len(development_positions)),
            "development_deaths": int(development_outcomes.sum()),
            "test_n": int(len(test_positions)),
            "test_deaths": int(test_outcomes.sum()),
        }
        if observed_counts != expected_counts:
            raise RuntimeError(
                "The default temporal-safe strict-OOF run reconstructed unexpected cohort "
                f"counts. Expected {expected_counts}; observed "
                f"{observed_counts}."
            )
        print("Current-run count audit: PASSED")

    return {
        "config": config,
        "data_path": data_path,
        "source": source,
        "cohort": cohort,
        "cohort_flow": cohort_flow,
        "target_column": target_column,
        "target": target,
        "development_positions": development_positions,
        "test_positions": test_positions,
        "development_predictions_file": development_file,
        "test_predictions_file": test_file,
    }


# =============================================================================
# VARIABLE NORMALIZATION
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
                "f",
                "female",
                "woman",
                "women",
                "false",
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
                "m",
                "male",
                "man",
                "men",
                "true",
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

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    for code, label in (
        SEX_NUMERIC_MAP.items()
    ):
        output.loc[
            numeric.eq(code)
        ] = label

    unresolved = (
        numeric.notna()
        & output.eq(
            "Missing/Other"
        )
    )

    if unresolved.any():
        unresolved_codes = sorted(
            numeric.loc[
                unresolved
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if STRICT_SEX_CODING:
            raise RuntimeError(
                "Unresolved numeric sex codes were detected: "
                f"{unresolved_codes}. Verify the data dictionary before "
                "generating a publication figure."
            )

        for code in unresolved_codes:
            output.loc[
                numeric.eq(code)
            ] = f"Code {code:g}"

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
        .map(parse_one)
        .astype(float)
    )

    return numeric.astype(float)


def normalize_urgency(
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


# =============================================================================
# TABLE BUILDERS
# =============================================================================

def count_table(
    group: pd.Series,
    outcome: pd.Series,
    variable: str,
    *,
    order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    table = (
        pd.DataFrame(
            {
                "group": group.astype(str),
                "outcome": outcome.to_numpy(),
            }
        )
        .groupby(
            "group",
            observed=True,
        )
        .agg(
            n=("outcome", "size"),
            deaths=("outcome", "sum"),
        )
        .reset_index()
    )

    n_total = len(outcome)

    table[
        "percent_of_cohort"
    ] = (
        100.0
        * table["n"]
        / n_total
    )

    table[
        "mortality_rate"
    ] = (
        table["deaths"]
        / table["n"].clip(
            lower=1
        )
    )

    table["variable"] = variable

    if order:
        order_map = {
            value: index
            for index, value in enumerate(
                order
            )
        }

        table["order"] = (
            table["group"]
            .map(order_map)
            .fillna(999)
        )

        table = (
            table.sort_values(
                [
                    "order",
                    "group",
                ]
            )
            .drop(
                columns="order"
            )
        )

    return table.reset_index(
        drop=True
    )


def mortality_table(
    group: pd.Series,
    outcome: pd.Series,
    variable: str,
    *,
    order: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = (
        "Missing/Other",
        "Unavailable",
    ),
) -> pd.DataFrame:
    temporary = pd.DataFrame(
        {
            "group": group.astype(str),
            "outcome": outcome.to_numpy(),
        }
    )

    temporary = temporary.loc[
        ~temporary[
            "group"
        ].isin(exclude)
    ].copy()

    grouped = (
        temporary.groupby(
            "group",
            observed=True,
        )
        .agg(
            n=("outcome", "size"),
            deaths=("outcome", "sum"),
        )
        .reset_index()
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for row in grouped.itertuples():
        lower, upper = wilson_interval(
            int(row.deaths),
            int(row.n),
        )

        rows.append(
            {
                "variable": variable,
                "group": str(
                    row.group
                ),
                "n": int(
                    row.n
                ),
                "deaths": int(
                    row.deaths
                ),
                "mortality_rate": (
                    row.deaths
                    / row.n
                ),
                "ci_lower_95": lower,
                "ci_upper_95": upper,            }
        )

    table = pd.DataFrame(rows)

    if order and not table.empty:
        order_map = {
            value: index
            for index, value in enumerate(
                order
            )
        }

        table["order"] = (
            table["group"]
            .map(order_map)
            .fillna(999)
        )

        table = (
            table.sort_values(
                [
                    "order",
                    "group",
                ]
            )
            .drop(
                columns="order"
            )
        )

    return table.reset_index(
        drop=True
    )


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_count_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    cmap,
) -> None:
    labels = table[
        "group"
    ].astype(str).tolist()

    values = table[
        "n"
    ].astype(float).to_numpy()

    colors = [
        cmap(value)
        for value in np.linspace(
            0.12,
            0.88,
            max(
                len(table),
                1,
            ),
        )
    ]

    positions = np.arange(
        len(table)
    )

    bars = axis.bar(
        positions,
        values,
        color=colors,
        edgecolor="none",
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels
    )
    axis.set_xlabel(
        xlabel
    )
    axis.set_ylabel(
        "Number of encounters"
    )
    axis.set_title(
        title,
        fontweight="semibold",
        loc="left",
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    maximum = max(
        float(
            values.max()
        ),
        1.0,
    )

    axis.set_ylim(
        0.0,
        maximum * 1.20,
    )

    for bar, row in zip(
        bars,
        table.itertuples(),
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height()
            + maximum
            * 0.025,
            (
                f"{int(row.n):,}\n"
                f"({row.percent_of_cohort:.1f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=10.5,
        )


def plot_mortality_bars(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    cmap,
    *,
    rotate: float = 0.0,
) -> None:
    if table.empty:
        axis.text(
            0.5,
            0.5,
            "Data unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_title(
            title,
            fontweight="bold",
            loc="left",
        )
        return

    labels = table[
        "group"
    ].astype(str).tolist()

    rates = (
        100.0
        * table[
            "mortality_rate"
        ].to_numpy(
            dtype=float
        )
    )

    lower = (
        100.0
        * table[
            "ci_lower_95"
        ].to_numpy(
            dtype=float
        )
    )

    upper = (
        100.0
        * table[
            "ci_upper_95"
        ].to_numpy(
            dtype=float
        )
    )

    y_error = np.vstack(
        [
            rates - lower,
            upper - rates,
        ]
    )

    colors = [
        cmap(value)
        for value in np.linspace(
            0.15,
            0.90,
            len(table),
        )
    ]

    positions = np.arange(
        len(table)
    )

    bars = axis.bar(
        positions,
        rates,
        yerr=y_error,
        capsize=3,
        color=colors,
        edgecolor="none",
        error_kw={
            "elinewidth": 1.3,
            "capthick": 1.3,
        },
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels,
        rotation=rotate,
        ha=(
            "right"
            if rotate
            else "center"
        ),
    )
    axis.set_xlabel(
        xlabel
    )
    axis.set_ylabel(
        "30-day mortality (%)"
    )
    axis.set_title(
        title,
        fontweight="semibold",
        loc="left",
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    maximum = max(
        float(
            upper.max()
        ),
        0.01,
    )

    axis.set_ylim(
        0.0,
        maximum * 1.30,
    )

    for bar, row, rate in zip(
        bars,
        table.itertuples(),
        rates,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            100.0
            * row.ci_upper_95
            + maximum
            * 0.035,
            (
                f"{rate:.2f}%\n"
                f"{int(row.deaths)}/{int(row.n)}"
            ),
            ha="center",
            va="bottom",
            fontsize=10.5,
        )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_figure2(
    run_dir: pathlib.Path,
    pipeline_module_path: pathlib.Path,
    data_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completion_file = (
        run_dir / "PARALLEL_RETRAINING_COMPLETE.json"
    )
    if completion_file.exists():
        completion = load_json(
            completion_file
        )
        if not bool(
            completion.get(
                "completed",
                False,
            )
        ):
            raise RuntimeError(
                "PARALLEL_RETRAINING_COMPLETE.json does not report a "
                "completed temporal-safe strict-OOF workflow."
            )

    configure_style()

    print("=" * 100)
    print("TEMPORAL-SAFE STRICT-OOF FIGURE 2")
    print("=" * 100)
    print("Run directory   :", run_dir)
    print("Pipeline module :", pipeline_module_path)
    print("Output directory:", output_dir)

    pipeline = load_pipeline_module(
        pipeline_module_path
    )

    reconstructed = reconstruct_current_cohort(
        pipeline,
        run_dir,
        data_path,
    )

    source = reconstructed[
        "source"
    ]
    cohort = reconstructed[
        "cohort"
    ]
    target = reconstructed[
        "target"
    ]
    target_column = reconstructed[
        "target_column"
    ]

    death_mask = target.eq(1)
    survival_mask = target.eq(0)

    n_total = len(cohort)
    deaths = int(
        death_mask.sum()
    )
    survivors = int(
        survival_mask.sum()
    )
    prevalence = float(
        death_mask.mean()
    )

    # -------------------------------------------------------------------------
    # Identify columns
    # -------------------------------------------------------------------------
    age_column = first_existing_column(
        cohort,
        [
            "age",
        ],
    )
    bmi_column = first_existing_column(
        cohort,
        [
            "bmi",
        ],
    )
    sex_column = first_existing_column(
        cohort,
        [
            "sex",
            "gender",
        ],
    )
    asa_column = first_existing_column(
        cohort,
        [
            "asa",
            "asa_status",
            "asa_class",
            "asa_ps",
        ],
    )
    urgency_column = first_existing_column(
        cohort,
        [
            "emop",
            "emergency",
            "emergency_status",
            "urgency",
        ],
    )
    department_column = first_existing_column(
        cohort,
        [
            "department",
            "surgical_department",
            "surgery_department",
            "department_name",
            "op_department",
            "optype",
            "surgery_type",
            "surgical_service",
        ],
    )

    age = (
        clean_numeric(
            cohort[age_column]
        )
        if age_column is not None
        else pd.Series(
            np.nan,
            index=cohort.index,
        )
    )

    bmi = (
        clean_numeric(
            cohort[bmi_column]
        )
        if bmi_column is not None
        else pd.Series(
            np.nan,
            index=cohort.index,
        )
    )

    bmi = bmi.where(
        bmi.between(
            10,
            80,
            inclusive="both",
        )
    )

    sex_group = (
        normalize_sex(
            cohort[sex_column]
        )
        if sex_column is not None
        else pd.Series(
            "Unavailable",
            index=cohort.index,
        )
    )

    asa_numeric = (
        parse_asa_numeric(
            cohort[asa_column]
        )
        if asa_column is not None
        else pd.Series(
            np.nan,
            index=cohort.index,
        )
    )

    urgency_group = (
        normalize_urgency(
            cohort[urgency_column]
        )
        if urgency_column is not None
        else pd.Series(
            "Unavailable",
            index=cohort.index,
        )
    )

    age_group = pd.Series(
        "Missing/Other",
        index=cohort.index,
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

    asa_group = pd.Series(
        "Missing/Other",
        index=cohort.index,
        dtype=object,
    )
    asa_group.loc[
        asa_numeric.isin(
            [1, 2]
        )
    ] = "I–II"
    asa_group.loc[
        asa_numeric.eq(3)
    ] = "III"
    asa_group.loc[
        asa_numeric.isin(
            [4, 5]
        )
    ] = "IV–V"

    # -------------------------------------------------------------------------
    # Reviewer-specific cohort checks
    # -------------------------------------------------------------------------
    raw_asa_column = first_existing_column(
        source,
        [
            "asa",
            "asa_status",
            "asa_class",
            "asa_ps",
        ],
    )

    raw_asa6_count = (
        int(
            parse_asa_numeric(
                source[
                    raw_asa_column
                ]
            )
            .eq(6)
            .sum()
        )
        if raw_asa_column is not None
        else 0
    )

    modeled_asa6_count = int(
        asa_numeric.eq(6).sum()
    )

    if modeled_asa6_count > 0:
        raise RuntimeError(
            f"{modeled_asa6_count} ASA 6 encounters remain in the "
            "temporal-safe strict-OOF modeled cohort."
        )

    emergency_case_count = int(
        urgency_group.eq(
            "Emergency"
        ).sum()
    )
    emergency_death_count = int(
        (
            urgency_group.eq(
                "Emergency"
            )
            & death_mask
        ).sum()
    )
    elective_case_count = int(
        urgency_group.eq(
            "Elective"
        ).sum()
    )
    elective_death_count = int(
        (
            urgency_group.eq(
                "Elective"
            )
            & death_mask
        ).sum()
    )

    print("\n" + "=" * 100)
    print("FINAL COHORT CHECKS")
    print("=" * 100)
    print(
        f"ASA 6 encounters in source data:  {raw_asa6_count:,}"
    )
    print(
        f"ASA 6 encounters in modeled data: {modeled_asa6_count:,}"
    )
    print(
        f"Emergency procedures: {emergency_case_count:,} "
        f"(deaths={emergency_death_count:,})"
    )
    print(
        f"Elective procedures:  {elective_case_count:,} "
        f"(deaths={elective_death_count:,})"
    )

    # -------------------------------------------------------------------------
    # Build tables
    # -------------------------------------------------------------------------
    sex_distribution = count_table(
        sex_group,
        target,
        "Sex",
        order=(
            SEX_GROUP_ORDER
            + [
                "Missing/Other",
                "Unavailable",
            ]
        ),
    )

    asa_distribution = count_table(
        asa_group,
        target,
        "ASA",
        order=(
            ASA_GROUP_ORDER
            + [
                "Missing/Other",
            ]
        ),
    )

    mortality_age = mortality_table(
        age_group,
        target,
        "Age",
        order=AGE_GROUP_ORDER,
    )

    mortality_asa = mortality_table(
        asa_group,
        target,
        "ASA",
        order=ASA_GROUP_ORDER,
    )

    mortality_urgency = mortality_table(
        urgency_group,
        target,
        "Urgency",
        order=URGENCY_ORDER,
    )

    mortality_sex = mortality_table(
        sex_group,
        target,
        "Sex",
        order=SEX_GROUP_ORDER,
    )

    department_raw: Optional[
        pd.Series
    ] = None

    if department_column is not None:
        department_raw = (
            cohort[
                department_column
            ]
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": "Missing/Other",
                    "nan": "Missing/Other",
                    "None": "Missing/Other",
                }
            )
        )

        department_counts = (
            department_raw.value_counts()
        )

        common_departments = (
            department_counts.loc[
                department_counts.ge(
                    MIN_DEPARTMENT_N
                )
                & ~department_counts.index.isin(
                    [
                        "Missing/Other",
                    ]
                )
            ]
            .head(
                TOP_DEPARTMENTS
            )
            .index
            .tolist()
        )

        department_group = (
            department_raw.where(
                department_raw.isin(
                    common_departments
                ),
                "Other/Not shown",
            )
        )

        mortality_department = mortality_table(
            department_group,
            target,
            "Surgical department",
            order=common_departments,
            exclude=(
                "Missing/Other",
                "Unavailable",
                "Other/Not shown",
            ),
        )

    else:
        common_departments = []
        mortality_department = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Diagnosis summaries and targeted eye/ear analysis
    # -------------------------------------------------------------------------
    diagnosis_columns = [
        column
        for column in cohort.columns
        if str(column).upper().startswith(
            "DIAG_"
        )
    ]

    diagnosis_rows: List[
        Dict[str, Any]
    ] = []

    if diagnosis_columns and deaths > 0:
        diagnosis_matrix = (
            cohort.loc[
                death_mask,
                diagnosis_columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
        )

        diagnosis_counts = (
            diagnosis_matrix.gt(0)
            .sum(axis=0)
            .sort_values(
                ascending=False
            )
            .head(
                TOP_DIAGNOSES
            )
        )

        for feature, count in (
            diagnosis_counts.items()
        ):
            diagnosis_rows.append(
                {
                    "feature": str(feature),
                    "display_label": compact_label(
                        re.sub(
                            r"^DIAG_",
                            "",
                            str(feature),
                            flags=re.IGNORECASE,
                        )
                    ),
                    "count_among_deaths": int(
                        count
                    ),
                    "percent_among_deaths": (
                        100.0
                        * int(count)
                        / deaths
                    ),
                }
            )

    diagnosis_summary = pd.DataFrame(
        diagnosis_rows
    )

    def normalized_token(
        name: str,
    ) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "",
            str(name).lower(),
        )

    eye_ear_candidates = [
        column
        for column in diagnosis_columns
        if (
            "eye"
            in normalized_token(
                column
            )
            and "ear"
            in normalized_token(
                column
            )
        )
    ]

    eye_ear_column = (
        eye_ear_candidates[0]
        if eye_ear_candidates
        else None
    )

    if eye_ear_column is not None:
        eye_ear_present = (
            pd.to_numeric(
                cohort[
                    eye_ear_column
                ],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
        )

        eye_ear_group = (
            eye_ear_present.map(
                {
                    False: "Absent",
                    True: "Present",
                }
            )
        )

        eye_ear_mortality = mortality_table(
            eye_ear_group,
            target,
            "Eye/Ear diagnosis",
            order=[
                "Absent",
                "Present",
            ],
            exclude=(),
        )

        eye_ear_prevalence = (
            pd.DataFrame(
                {
                    "outcome": np.where(
                        death_mask,
                        "Death",
                        "Survivor",
                    ),
                    "eye_ear_status": (
                        eye_ear_group
                    ),
                }
            )
            .groupby(
                [
                    "outcome",
                    "eye_ear_status",
                ],
                observed=True,
            )
            .size()
            .rename("n")
            .reset_index()
        )

        outcome_totals = (
            eye_ear_prevalence.groupby(
                "outcome"
            )["n"]
            .transform("sum")
        )

        eye_ear_prevalence[
            "percent_within_outcome"
        ] = (
            100.0
            * eye_ear_prevalence["n"]
            / outcome_totals
        )

        stratified_rows: List[
            Dict[str, Any]
        ] = []

        stratifiers: Dict[
            str,
            pd.Series,
        ] = {
            "ASA physical status": asa_group,
            "Urgency": urgency_group,
        }

        if department_raw is not None:
            stratifiers[
                "Surgical department"
            ] = department_raw

        for (
            stratifier_name,
            stratifier_values,
        ) in stratifiers.items():
            temporary = pd.DataFrame(
                {
                    "stratum": (
                        stratifier_values.astype(
                            str
                        )
                    ),
                    "eye_ear_status": (
                        eye_ear_group.astype(
                            str
                        )
                    ),
                    "outcome": target.to_numpy(),
                }
            )

            temporary = temporary.loc[
                ~temporary[
                    "stratum"
                ].isin(
                    [
                        "Missing/Other",
                        "Unavailable",
                        "nan",
                        "None",
                        "",
                    ]
                )
            ].copy()

            grouped = (
                temporary.groupby(
                    [
                        "stratum",
                        "eye_ear_status",
                    ],
                    observed=True,
                )
                .agg(
                    n=(
                        "outcome",
                        "size",
                    ),
                    deaths=(
                        "outcome",
                        "sum",
                    ),
                )
                .reset_index()
            )

            for row in (
                grouped.itertuples()
            ):
                lower, upper = wilson_interval(
                    int(row.deaths),
                    int(row.n),
                )

                stratified_rows.append(
                    {
                        "stratifier": stratifier_name,
                        "stratum": str(
                            row.stratum                        ),
                        "eye_ear_status": str(
                            row.eye_ear_status
                        ),
                        "n": int(
                            row.n
                        ),
                        "deaths": int(
                            row.deaths
                        ),
                        "mortality_rate": (
                            row.deaths
                            / row.n
                        ),
                        "ci_lower_95": lower,
                        "ci_upper_95": upper,
                    }
                )

        eye_ear_stratified = pd.DataFrame(
            stratified_rows
        )

    else:
        eye_ear_mortality = pd.DataFrame()
        eye_ear_prevalence = pd.DataFrame()
        eye_ear_stratified = pd.DataFrame()

        print(
            "Warning: no diagnosis feature containing both 'eye' and 'ear' "
            "was found. The supplementary sensitivity figure will be skipped."
        )

    # -------------------------------------------------------------------------
    # Main 3×3 figure
    # -------------------------------------------------------------------------
    cmap = cm.get_cmap(
        CMAP_NAME
    )

    survivor_color = cmap(0.20)
    death_color = cmap(0.88)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 16),
    )

    (
        (axis_a, axis_b, axis_c),
        (axis_d, axis_e, axis_f),
        (axis_g, axis_h, axis_i),
    ) = axes

    fig.suptitle(
        "Cohort characteristics and 30-day postoperative mortality patterns",
        fontsize=20,
        fontweight="bold",
        y=0.992,
    )

    # Keep run provenance and cohort-flow details in the manuscript caption
    # rather than inside the plotting area.

    # Panel A — Age
    if age_column is not None:
        age_survivors = age.loc[
            survival_mask
        ].dropna()

        age_deaths = age.loc[
            death_mask
        ].dropna()

        combined_age = pd.concat(
            [
                age_survivors,
                age_deaths,
            ]
        )

        if len(combined_age) > 0:
            lower_age = max(
                0.0,
                np.floor(
                    combined_age.quantile(
                        0.005
                    )
                    / 5.0
                )
                * 5.0,
            )

            upper_age = (
                np.ceil(
                    combined_age.quantile(
                        0.995
                    )
                    / 5.0
                )
                * 5.0
            )

            bins = np.arange(
                lower_age,
                upper_age + 5.0,
                5.0,
            )

            axis_a.hist(
                age_survivors,
                bins=bins,
                density=True,
                alpha=0.55,
                color=survivor_color,
                label="Survived 30 days",
            )

            axis_a.hist(
                age_deaths,
                bins=bins,
                density=True,
                alpha=0.68,
                color=death_color,
                label="Died within 30 days",
            )

            axis_a.axvline(
                age_survivors.median(),
                color=survivor_color,
                linestyle="--",
                linewidth=1.5,
            )

            axis_a.axvline(
                age_deaths.median(),
                color=death_color,
                linestyle="--",
                linewidth=1.5,
            )

        axis_a.set_xlabel(
            "Age (years)"
        )
        axis_a.set_ylabel(
            "Density"
        )
        axis_a.legend(
            frameon=False,
            fontsize=11,
        )

    else:
        axis_a.text(
            0.5,
            0.5,
            "Age unavailable",
            transform=axis_a.transAxes,
            ha="center",
            va="center",
        )

    axis_a.set_title(
        "Age",
        fontweight="semibold",
        loc="left",
    )
    axis_a.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    # Panel B — BMI
    if bmi_column is not None:
        bmi_survivors = bmi.loc[
            survival_mask
        ].dropna()

        bmi_deaths = bmi.loc[
            death_mask
        ].dropna()

        combined_bmi = pd.concat(
            [
                bmi_survivors,
                bmi_deaths,
            ]
        )

        if len(combined_bmi) > 0:
            lower_bmi = max(
                10.0,
                np.floor(
                    combined_bmi.quantile(
                        0.005
                    )
                    / 2.0
                )
                * 2.0,
            )

            upper_bmi = min(
                80.0,
                np.ceil(
                    combined_bmi.quantile(
                        0.995
                    )
                    / 2.0
                )
                * 2.0,
            )

            bins = np.arange(
                lower_bmi,
                upper_bmi + 2.0,
                2.0,
            )

            axis_b.hist(
                bmi_survivors,
                bins=bins,
                density=True,
                alpha=0.55,
                color=survivor_color,
                label="Survived 30 days",
            )

            axis_b.hist(
                bmi_deaths,
                bins=bins,
                density=True,
                alpha=0.68,
                color=death_color,
                label="Died within 30 days",
            )

            axis_b.axvline(
                bmi_survivors.median(),
                color=survivor_color,
                linestyle="--",
                linewidth=1.5,
            )

            axis_b.axvline(
                bmi_deaths.median(),
                color=death_color,
                linestyle="--",
                linewidth=1.5,
            )

        axis_b.set_xlabel(
            "Body mass index (kg/m²)"
        )
        axis_b.set_ylabel(
            "Density"
        )
        axis_b.legend(
            frameon=False,
            fontsize=11,
        )

    else:
        axis_b.text(
            0.5,
            0.5,
            "BMI unavailable",
            transform=axis_b.transAxes,
            ha="center",
            va="center",
        )

    axis_b.set_title(
        "Body mass index",
        fontweight="semibold",
        loc="left",
    )
    axis_b.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    # Panels C–I
    plot_count_bars(
        axis_c,
        sex_distribution,
        "Sex",
        "Sex",
        cmap,
    )

    plot_count_bars(
        axis_d,
        asa_distribution,
        "ASA physical status",
        "ASA physical status",
        cmap,
    )

    plot_mortality_bars(
        axis_e,
        mortality_age,
        "30-day mortality by age",
        "Age (years)",
        cmap,
    )

    plot_mortality_bars(
        axis_f,
        mortality_asa,
        "30-day mortality by ASA status",
        "ASA physical status",
        cmap,
    )

    plot_mortality_bars(
        axis_g,
        mortality_urgency,
        "30-day mortality by urgency",
        "Urgency",
        cmap,
    )

    plot_mortality_bars(
        axis_h,
        mortality_department,
        "30-day mortality by surgical department",
        "Surgical department",
        cmap,
        rotate=30.0,
    )

    plot_mortality_bars(
        axis_i,
        mortality_sex,
        "30-day mortality by sex",
        "Sex",
        cmap,
    )

    for axis, label in zip(
        axes.flatten(),
        list("ABCDEFGHI"),
    ):
        add_panel_label(
            axis,
            label,
        )

    fig.tight_layout(
        rect=(
            0.025,
            0.025,
            0.99,
            0.955,
        ),
        h_pad=3.2,
        w_pad=2.8,
    )

    main_figure_path = (
        output_dir
        / f"{FIGURE_BASENAME}.png"
    )

    save_figure_all_formats(
        fig,
        main_figure_path,
    )

    plt.show()
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Supplementary eye/ear sensitivity figure
    # -------------------------------------------------------------------------
    eye_ear_figure_path: Optional[
        pathlib.Path
    ] = None

    if not eye_ear_mortality.empty:
        fig_eye, axis_eye = plt.subplots(
            figsize=(8.5, 6.5)
        )

        plot_mortality_bars(
            axis_eye,
            eye_ear_mortality,
            "Eye/ear diagnosis sensitivity analysis",
            "Eye/ear diagnosis",
            cmap,
        )

        death_present = (
            eye_ear_prevalence.loc[
                eye_ear_prevalence[
                    "outcome"
                ].eq("Death")
                & eye_ear_prevalence[
                    "eye_ear_status"
                ].eq("Present"),
                "percent_within_outcome",
            ]
        )

        survivor_present = (
            eye_ear_prevalence.loc[
                eye_ear_prevalence[
                    "outcome"
                ].eq("Survivor")
                & eye_ear_prevalence[
                    "eye_ear_status"
                ].eq("Present"),
                "percent_within_outcome",
            ]
        )

        death_prevalence = (
            float(
                death_present.iloc[0]
            )
            if len(death_present)
            else np.nan
        )

        survivor_prevalence = (
            float(
                survivor_present.iloc[0]
            )
            if len(survivor_present)
            else np.nan
        )

        axis_eye.text(
            0.98,
            0.96,
            (
                f"Variable: {eye_ear_column}\n"
                f"Present among deaths: "
                f"{death_prevalence:.2f}%\n"
                f"Present among survivors: "
                f"{survivor_prevalence:.2f}%"
            ),
            transform=axis_eye.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color="dimgray",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.90,
            },
        )

        fig_eye.text(
            0.06,
            0.02,
            (
                "Error bars are Wilson 95% confidence intervals. This "
                "descriptive analysis should be interpreted with the saved "
                "ASA-, urgency-, and department-stratified tables."
            ),
            ha="left",
            va="bottom",
            fontsize=10,
            color="dimgray",
        )

        fig_eye.tight_layout(
            rect=(
                0.03,
                0.06,
                0.99,
                0.98,
            )
        )

        eye_ear_figure_path = (
            output_dir
            / "Supplementary_EyeEar_Diagnosis_Sensitivity.png"
        )

        save_figure_all_formats(
            fig_eye,
            eye_ear_figure_path,
        )

        plt.show()
        plt.close(fig_eye)

    # -------------------------------------------------------------------------
    # Numerical outputs
    # -------------------------------------------------------------------------
    cohort_summary = pd.DataFrame(
        [
            {
                "analytical_cohort_n": n_total,
                "deaths_30d": deaths,
                "survivors_30d": survivors,
                "mortality_prevalence": prevalence,
                "development_n": len(
                    reconstructed[
                        "development_positions"
                    ]
                ),
                "development_deaths": int(
                    target.iloc[
                        reconstructed[
                            "development_positions"
                        ]
                    ].sum()
                ),
                "test_n": len(
                    reconstructed[
                        "test_positions"
                    ]
                ),
                "test_deaths": int(
                    target.iloc[
                        reconstructed[
                            "test_positions"
                        ]
                    ].sum()
                ),
                "source_n": len(source),
                "raw_asa6_count": raw_asa6_count,
                "modeled_asa6_count": modeled_asa6_count,
                "emergency_case_count": emergency_case_count,
                "emergency_death_count": emergency_death_count,
                "elective_case_count": elective_case_count,
                "elective_death_count": elective_death_count,
            }
        ]
    )

    distribution_summary = pd.concat(
        [
            sex_distribution,
            asa_distribution,
        ],
        ignore_index=True,
        sort=False,
    )

    mortality_summary = pd.concat(
        [
            mortality_age,
            mortality_asa,
            mortality_urgency,
            mortality_department,
            mortality_sex,
        ],
        ignore_index=True,
        sort=False,
    )

    continuous_rows: List[
        Dict[str, Any]
    ] = []

    for name, values in [
        (
            "Age, overall",
            age,
        ),
        (
            "Age, survivors",
            age.loc[
                survival_mask
            ],
        ),
        (
            "Age, deaths",
            age.loc[
                death_mask
            ],
        ),
        (
            "BMI, overall",
            bmi,
        ),
        (
            "BMI, survivors",
            bmi.loc[
                survival_mask
            ],
        ),
        (
            "BMI, deaths",
            bmi.loc[
                death_mask
            ],
        ),
    ]:
        values = values.dropna()

        if len(values) == 0:
            continue

        continuous_rows.append(
            {
                "variable": name,
                "n_nonmissing": len(
                    values
                ),
                "mean": float(
                    values.mean()
                ),
                "sd": float(
                    values.std(
                        ddof=1
                    )
                ),
                "median": float(
                    values.median()
                ),
                "q1": float(
                    values.quantile(
                        0.25
                    )
                ),
                "q3": float(
                    values.quantile(
                        0.75
                    )
                ),
                "minimum": float(
                    values.min()
                ),
                "maximum": float(
                    values.max()
                ),
            }
        )

    continuous_summary = pd.DataFrame(
        continuous_rows
    )

    cohort_summary_path = (
        output_dir
        / "Figure2_cohort_summary.csv"
    )
    distribution_path = (
        output_dir
        / "Figure2_distribution_tables.csv"
    )
    mortality_path = (
        output_dir
        / "Figure2_mortality_rates_with_Wilson_CIs.csv"
    )
    diagnosis_path = (
        output_dir
        / "Figure2_top_diagnoses_among_deaths.csv"
    )
    continuous_path = (
        output_dir
        / "Figure2_continuous_characteristics.csv"
    )
    reviewer_checks_path = (
        output_dir
        / "Figure2_final_cohort_checks.csv"
    )
    eye_ear_mortality_path = (
        output_dir
        / "Figure2_EyeEar_mortality.csv"
    )
    eye_ear_prevalence_path = (
        output_dir
        / "Figure2_EyeEar_prevalence_by_outcome.csv"
    )
    eye_ear_stratified_path = (
        output_dir
        / "Figure2_EyeEar_stratified_sensitivity.csv"
    )
    summary_path = (
        output_dir
        / "Figure2_temporal_safe_strict_oof_summary.json"
    )
    manuscript_path = (
        output_dir
        / "Figure2_manuscript_numerics.txt"
    )

    cohort_summary.to_csv(
        cohort_summary_path,
        index=False,
    )
    distribution_summary.to_csv(
        distribution_path,
        index=False,
    )
    mortality_summary.to_csv(
        mortality_path,
        index=False,
    )
    diagnosis_summary.to_csv(
        diagnosis_path,
        index=False,
    )
    continuous_summary.to_csv(
        continuous_path,
        index=False,
    )
    cohort_summary.to_csv(
        reviewer_checks_path,
        index=False,
    )
    eye_ear_mortality.to_csv(
        eye_ear_mortality_path,
        index=False,
    )
    eye_ear_prevalence.to_csv(
        eye_ear_prevalence_path,
        index=False,
    )
    eye_ear_stratified.to_csv(
        eye_ear_stratified_path,
        index=False,
    )

    manuscript_lines = [
        "FIGURE 2 — TEMPORAL-SAFE STRICT-OOF MANUSCRIPT NUMERICAL RESULTS",
        "=" * 82,
        "",
        (
            f"The temporal-safe strict-OOF modeled cohort included {n_total:,} encounters "
            f"and {deaths:,} deaths within 30 days "
            f"({100.0 * prevalence:.3f}%)."
        ),
        (
            f"The development cohort included "
            f"{len(reconstructed['development_positions']):,} encounters "
            f"with {int(target.iloc[reconstructed['development_positions']].sum()):,} "
            f"deaths; the held-out test cohort included "
            f"{len(reconstructed['test_positions']):,} encounters with "
            f"{int(target.iloc[reconstructed['test_positions']].sum()):,} deaths."
        ),
        (
            f"ASA physical status 6 encounters remaining after cohort "
            f"preparation: {modeled_asa6_count}. Source-data ASA 6 count: "
            f"{raw_asa6_count}."
        ),
        (
            f"Emergency procedures were included: "
            f"{emergency_case_count:,} encounters with "
            f"{emergency_death_count:,} deaths. Elective procedures included "
            f"{elective_case_count:,} encounters with "
            f"{elective_death_count:,} deaths."
        ),
        "",
        "CONTINUOUS CHARACTERISTICS",
    ]

    for row in continuous_summary.itertuples():
        manuscript_lines.append(
            (
                f"{row.variable}: N={int(row.n_nonmissing):,}; "
                f"mean ± SD {row.mean:.2f} ± {row.sd:.2f}; "
                f"median (IQR) {row.median:.2f} "
                f"({row.q1:.2f}–{row.q3:.2f})."
            )
        )

    manuscript_lines.extend(
        [
            "",
            "MORTALITY RATES",
        ]
    )

    for row in mortality_summary.itertuples():
        manuscript_lines.append(
            (
                f"{row.variable}, {row.group}: "
                f"{100.0 * row.mortality_rate:.3f}% "
                f"(95% CI {100.0 * row.ci_lower_95:.3f}–"
                f"{100.0 * row.ci_upper_95:.3f}); "
                f"{int(row.deaths)}/{int(row.n)} deaths."
            )
        )

    if not diagnosis_summary.empty:
        manuscript_lines.extend(
            [
                "",
                "MOST COMMON DIAGNOSES AMONG DEATHS",
            ]
        )

        for row in diagnosis_summary.itertuples():
            manuscript_lines.append(
                (
                    f"{row.display_label}: "
                    f"{int(row.count_among_deaths)}/{deaths} deaths "
                    f"({row.percent_among_deaths:.2f}%)."
                )
            )

    if not eye_ear_mortality.empty:
        manuscript_lines.extend(
            [
                "",
                "TARGETED EYE/EAR DIAGNOSIS SENSITIVITY",
            ]
        )

        for row in eye_ear_mortality.itertuples():
            manuscript_lines.append(
                (
                    f"Eye/ear diagnosis {row.group}: "
                    f"{100.0 * row.mortality_rate:.3f}% mortality "
                    f"(95% CI {100.0 * row.ci_lower_95:.3f}–"
                    f"{100.0 * row.ci_upper_95:.3f}); "
                    f"{int(row.deaths)}/{int(row.n)} deaths."
                )
            )

    manuscript_path.write_text(
        "\n".join(
            manuscript_lines
        ),
        encoding="utf-8",
    )

    eye_ear_retraining_directory = (
        run_dir / "eye_ear_feature_retraining"
    )
    eye_ear_retraining_files = (
        sorted(
            str(path)
            for path in eye_ear_retraining_directory.glob("*")
            if path.is_file()
        )
        if eye_ear_retraining_directory.exists()
        else []
    )

    summary = {
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
        "target_column": target_column,
        "analytical_cohort_n": int(
            n_total
        ),
        "deaths_30d": int(
            deaths
        ),
        "survivors_30d": int(
            survivors
        ),
        "mortality_prevalence": prevalence,
        "development_n": int(
            len(
                reconstructed[
                    "development_positions"
                ]
            )
        ),
        "development_deaths": int(
            target.iloc[
                reconstructed[
                    "development_positions"
                ]
            ].sum()
        ),
        "test_n": int(
            len(
                reconstructed[
                    "test_positions"
                ]
            )
        ),
        "test_deaths": int(
            target.iloc[
                reconstructed[
                    "test_positions"
                ]
            ].sum()
        ),
        "development_predictions_file": str(
            reconstructed[
                "development_predictions_file"
            ]
        ),
        "test_predictions_file": str(
            reconstructed[
                "test_predictions_file"
            ]
        ),
        "age_column": age_column,
        "bmi_column": bmi_column,
        "sex_column": sex_column,
        "asa_column": asa_column,
        "urgency_column": urgency_column,
        "department_column": department_column,
        "raw_asa6_count": int(
            raw_asa6_count
        ),
        "modeled_asa6_count": int(
            modeled_asa6_count
        ),
        "emergency_case_count": int(
            emergency_case_count
        ),
        "emergency_death_count": int(
            emergency_death_count
        ),
        "elective_case_count": int(
            elective_case_count
        ),
        "elective_death_count": int(
            elective_death_count
        ),
        "departments_displayed": (
            common_departments
        ),
        "diagnoses_displayed": (
            diagnosis_summary[
                "feature"
            ].tolist()
            if not diagnosis_summary.empty
            else []
        ),
        "eye_ear_column": eye_ear_column,
        "main_figure": str(
            main_figure_path
        ),
        "eye_ear_sensitivity_figure": (
            None
            if eye_ear_figure_path is None
            else str(
                eye_ear_figure_path
            )
        ),
        "eye_ear_feature_retraining_directory": (
            str(eye_ear_retraining_directory)
            if eye_ear_retraining_directory.exists()
            else None
        ),
        "eye_ear_feature_retraining_files": (
            eye_ear_retraining_files
        ),
        "split_verification": {
            "development_and_test_unique": True,
            "development_and_test_nonoverlapping": True,
            "all_modeled_rows_accounted_for": True,
            "saved_outcomes_match_reconstructed_outcomes": True,
        },
        "interpretation_note": (
            "Figure 2 is descriptive and uses the complete corrected modeled "
            "cohort. The eye/ear cohort comparison is unadjusted and should "
            "not be conflated with the separate temporal-safe strict-OOF feature-removal "
            "retraining analysis."
        ),
    }

    save_json(
        summary,
        summary_path,
    )

    print("\n" + "=" * 100)
    print("TEMPORAL-SAFE STRICT-OOF FIGURE 2 COMPLETE")
    print("=" * 100)
    print("Main figure:", main_figure_path)
    print("Output directory:", output_dir)

    if eye_ear_figure_path is not None:
        print(
            "Supplementary eye/ear figure:",
            eye_ear_figure_path,
        )

    print("\nContinuous characteristics:")
    print(
        continuous_summary.round(
            4
        ).to_string(
            index=False
        )
    )

    print("\nMortality rates:")
    print(        mortality_summary.round(
            6
        ).to_string(
            index=False
        )
    )

    print("\nManuscript-ready numerics:")
    print(
        "\n".join(
            manuscript_lines
        )
    )


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
            "Generate the current temporal-safe strict-OOF Figure 2 cohort "
            "characteristics and mortality-pattern figure."
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

    run_figure2(
        run_dir=args.run_dir,
        pipeline_module_path=args.pipeline_module,
        data_path=args.data_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()