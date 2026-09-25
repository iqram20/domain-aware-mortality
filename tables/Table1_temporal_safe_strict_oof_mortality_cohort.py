#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

TABLE 1 — FINAL TEMPORAL-SAFE STRICT-OOF MORTALITY COHORT CHARACTERISTICS

================================================================



Completed run

-------------

/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/

darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653



Current pipeline

----------------

/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Code/New/

Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py



This script reconstructs the exact corrected cohort used in the completed

strict-OOF run, verifies the saved development and held-out test partitions,

and produces Table 1 stratified by 30-day mortality outcome.

"""



from __future__ import annotations



import importlib.util

import json

import pathlib

import sys

from typing import Any, Dict, Iterable, List, Optional, Sequence



import numpy as np

import pandas as pd





# =============================================================================

# FINAL TEMPORAL-SAFE STRICT-OOF PATHS

# =============================================================================



PIPELINE_MODULE = pathlib.Path(

    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Code/New/"

    "Mortality_DARN_full_pipe_v6_temporal_safe_strict_oof.py"

)



RUN_DIR = pathlib.Path(

    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"

    "darn_cv_baseline_runs/v6_temporal_safe_strict_oof_parallel/run_20260919_115653"

)



DATA_PATH = pathlib.Path(

    "/athena/madelab/scratch/iqh4001/VitalDB/Inspire/Results/"

    "Final_Datasets/df_final_mortality_2026-05-24_12-44_temporal_safe.csv"

)



OUTDIR = RUN_DIR / "tables"

OUTDIR.mkdir(parents=True, exist_ok=True)



TABLE1_PATH = OUTDIR / "Table1_Mortality_Cohort_StrictOOF.csv"

COHORT_FLOW_PATH = OUTDIR / "Table1_StrictOOF_Cohort_Flow.csv"

SPLIT_AUDIT_PATH = OUTDIR / "Table1_StrictOOF_Split_Audit.csv"



EXPECTED_CURRENT_RUN_COUNTS = {

    "cohort_n": 99_834,

    "cohort_deaths": 235,

    "development_n": 79_867,

    "development_deaths": 188,

    "test_n": 19_967,

    "test_deaths": 47,

}





# =============================================================================

# FILE, MODULE, AND CONFIGURATION HELPERS

# =============================================================================



def require_file(path: pathlib.Path) -> pathlib.Path:

    if not path.exists():

        raise FileNotFoundError(

            f"Required current-run file not found:\n{path}"

        )

    return path





def load_json(path: pathlib.Path) -> Dict[str, Any]:

    with open(require_file(path), "r", encoding="utf-8") as handle:

        return json.load(handle)





def load_pipeline_module(path: pathlib.Path):

    require_file(path)



    spec = importlib.util.spec_from_file_location(

        "mortality_darn_v6_table1_module",

        path,

    )



    if spec is None or spec.loader is None:

        raise ImportError(f"Could not import pipeline module:\n{path}")



    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module





def first_existing_file(

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



        matched = lower_map.get(candidate.lower())

        if matched is not None:

            return matched



    return None





def prediction_position_column(prediction_table: pd.DataFrame) -> str:

    column = first_existing_column(

        prediction_table,

        [

            "final_cohort_row_position",

            "original_row_position",

        ],

    )



    if column is None:

        raise KeyError(

            "Prediction table lacks a cohort row-position column. Expected "

            "'final_cohort_row_position' or 'original_row_position'."

        )



    return column





# =============================================================================

# EXACT FINAL COHORT AND SPLIT RECONSTRUCTION

# =============================================================================



def reconstruct_current_run():

    pipeline = load_pipeline_module(PIPELINE_MODULE)

    config = load_json(RUN_DIR / "config.json")



    configured_data_path = pathlib.Path(

        str(config.get("data_path", DATA_PATH))

    )

    data_path = (

        configured_data_path

        if configured_data_path.exists()

        else DATA_PATH

    )



    source = pd.read_csv(

        require_file(data_path),

        low_memory=False,

    )



    target_column = str(

        config.get("target_column", "mortality_30d")

    )



    if target_column not in source.columns:

        raise KeyError(

            f"Target column '{target_column}' is absent from the source data."

        )



    cohort, cohort_flow = pipeline.apply_reviewer_cohort_exclusions(

        source,

        target_column=target_column,

        exclude_asa6=bool(config.get("exclude_asa6", True)),

    )

    cohort = cohort.reset_index(drop=True)



    development_file = first_existing_file(

        [

            RUN_DIR / "development_oof_predictions.csv",

            RUN_DIR / "development_predictions.csv",

        ],

        "the strict-OOF development prediction table",

    )

    test_file = require_file(

        RUN_DIR / "test_predictions_all_models.csv"

    )



    development_predictions = pd.read_csv(development_file)

    test_predictions = pd.read_csv(test_file)



    development_position_col = prediction_position_column(

        development_predictions

    )

    test_position_col = prediction_position_column(

        test_predictions

    )



    development_positions = pd.to_numeric(

        development_predictions[development_position_col],

        errors="raise",

    ).astype(int).to_numpy()



    test_positions = pd.to_numeric(

        test_predictions[test_position_col],

        errors="raise",

    ).astype(int).to_numpy()



    for name, positions in (

        ("development", development_positions),

        ("holdout", test_positions),

    ):

        if np.any(positions < 0) or np.any(positions >= len(cohort)):

            raise IndexError(

                f"Saved {name} row positions fall outside the corrected cohort."

            )



        if len(np.unique(positions)) != len(positions):

            raise RuntimeError(

                f"Saved {name} row positions contain duplicates."

            )



    overlap = np.intersect1d(

        development_positions,

        test_positions,

    )

    if len(overlap) > 0:

        raise RuntimeError(

            "Development and held-out test row positions overlap."

        )



    represented_positions = np.concatenate(

        [development_positions, test_positions]

    )

    expected_positions = np.arange(len(cohort), dtype=int)



    missing_positions = np.setdiff1d(

        expected_positions,

        represented_positions,

    )

    unexpected_positions = np.setdiff1d(

        represented_positions,

        expected_positions,

    )



    if len(missing_positions) > 0 or len(unexpected_positions) > 0:

        raise RuntimeError(

            "Saved development and test partitions do not exhaustively "

            "reconstruct the corrected modeled cohort."

        )



    target = pd.to_numeric(

        cohort[target_column],

        errors="coerce",

    )



    if target.isna().any():

        raise ValueError(

            "The corrected cohort contains missing 30-day mortality labels."

        )



    target = target.astype(int)

    if not set(target.unique()).issubset({0, 1}):

        raise ValueError(

            "The corrected mortality target contains values other than 0 and 1."

        )



    development_outcomes = target.iloc[

        development_positions

    ].to_numpy()

    test_outcomes = target.iloc[

        test_positions

    ].to_numpy()



    if "y_true" in development_predictions.columns:

        saved = pd.to_numeric(

            development_predictions["y_true"],

            errors="raise",

        ).astype(int).to_numpy()

        if not np.array_equal(development_outcomes, saved):

            raise RuntimeError(

                "Development outcomes do not match saved OOF predictions."

            )



    if "y_true" in test_predictions.columns:

        saved = pd.to_numeric(

            test_predictions["y_true"],

            errors="raise",

        ).astype(int).to_numpy()

        if not np.array_equal(test_outcomes, saved):

            raise RuntimeError(

                "Holdout outcomes do not match saved test predictions."

            )



    observed_counts = {

        "cohort_n": int(len(cohort)),

        "cohort_deaths": int(target.sum()),

        "development_n": int(len(development_positions)),

        "development_deaths": int(development_outcomes.sum()),

        "test_n": int(len(test_positions)),

        "test_deaths": int(test_outcomes.sum()),

    }



    if observed_counts != EXPECTED_CURRENT_RUN_COUNTS:

        raise RuntimeError(

            "Unexpected final strict-OOF cohort counts.\n"

            f"Expected: {EXPECTED_CURRENT_RUN_COUNTS}\n"

            f"Observed: {observed_counts}"

        )



    split_audit = pd.DataFrame(

        [

            {

                "Cohort": "Development",

                "Encounters": observed_counts["development_n"],

                "Deaths": observed_counts["development_deaths"],

                "Mortality_prevalence_percent": (

                    100.0

                    * observed_counts["development_deaths"]

                    / observed_counts["development_n"]

                ),

            },

            {

                "Cohort": "Held-out test",

                "Encounters": observed_counts["test_n"],

                "Deaths": observed_counts["test_deaths"],

                "Mortality_prevalence_percent": (

                    100.0

                    * observed_counts["test_deaths"]

                    / observed_counts["test_n"]

                ),

            },

            {

                "Cohort": "Overall modeled cohort",

                "Encounters": observed_counts["cohort_n"],

                "Deaths": observed_counts["cohort_deaths"],

                "Mortality_prevalence_percent": (

                    100.0

                    * observed_counts["cohort_deaths"]

                    / observed_counts["cohort_n"]

                ),

            },

        ]

    )



    return {

        "pipeline": pipeline,

        "config": config,

        "data_path": data_path,

        "source": source,

        "cohort": cohort,

        "cohort_flow": cohort_flow,

        "split_audit": split_audit,

        "target_column": target_column,

        "target": target,

        "development_positions": development_positions,

        "test_positions": test_positions,

        "development_predictions_file": development_file,

        "test_predictions_file": test_file,

    }





# =============================================================================

# TABLE-SUMMARY HELPERS

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



    # Current INSPIRE coding used in the analysis:

    # 0 / False = Female; 1 / True = Male.

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





def normalize_emergency(series: pd.Series) -> tuple[pd.Series, pd.Series]:

    raw = series.astype(str).str.strip().str.lower()

    values = pd.to_numeric(series, errors="coerce")



    positive_text = {

        "true", "yes", "y", "emergency", "emergent", "urgent"

    }

    negative_text = {

        "false", "no", "n", "elective", "scheduled"

    }



    flag = values.eq(1) | raw.isin(positive_text)

    valid = values.isin([0, 1]) | raw.isin(positive_text | negative_text)

    return flag, valid





def derive_icu_status(

    frame: pd.DataFrame,

) -> tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:

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

) -> Optional[str]:

    column = first_existing_column(cohort, candidates)

    if column is None:

        return None



    rows.append(

        [

            label,

            continuous_summary(cohort[column]),

            continuous_summary(survivors[column]),

            continuous_summary(deaths[column]),

        ]

    )

    return column





# =============================================================================

# BUILD TABLE 1

# =============================================================================



def build_table1(reconstructed: Dict[str, Any]) -> pd.DataFrame:

    cohort = reconstructed["cohort"].copy()

    target_column = reconstructed["target_column"]



    cohort[target_column] = pd.to_numeric(

        cohort[target_column],

        errors="raise",

    ).astype(int)



    survivors = cohort.loc[cohort[target_column].eq(0)].copy()

    deaths = cohort.loc[cohort[target_column].eq(1)].copy()



    rows: List[List[str]] = [

        [

            "Surgical encounters, n",

            f"{len(cohort):,}",

            f"{len(survivors):,}",

            f"{len(deaths):,}",

        ]

    ]



    patient_id_column = first_existing_column(

        cohort,

        [

            "subject_id",

            "subjectid",

            "patient_id",

            "patientid",

        ],

    )



    if patient_id_column is not None:

        rows.insert(

            0,

            [

                "Unique patients, n",

                f"{cohort[patient_id_column].nunique(dropna=True):,}",

                f"{survivors[patient_id_column].nunique(dropna=True):,}",

                f"{deaths[patient_id_column].nunique(dropna=True):,}",

            ],

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

        sex_valid = sex_group.ne("Missing/Other")



        rows.append(

            [

                "Male sex, n (%)",

                binary_summary(sex_group.eq("Male"), sex_valid),

                binary_summary(

                    sex_group.loc[survivors.index].eq("Male"),

                    sex_valid.loc[survivors.index],

                ),

                binary_summary(

                    sex_group.loc[deaths.index].eq("Male"),

                    sex_valid.loc[deaths.index],

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

        asa_numeric = parse_asa(cohort[asa_column])



        for label, mask in (

            ("ASA I–II, n (%)", asa_numeric.isin([1, 2])),

            ("ASA III, n (%)", asa_numeric.eq(3)),

            ("ASA IV–V, n (%)", asa_numeric.isin([4, 5])),

            (

                "ASA missing/other, n (%)",

                ~asa_numeric.isin([1, 2, 3, 4, 5]),

            ),

        ):

            rows.append(

                [

                    label,

                    binary_summary(mask),

                    binary_summary(mask.loc[survivors.index]),

                    binary_summary(mask.loc[deaths.index]),

                ]

            )



        if asa_numeric.eq(6).any():

            raise RuntimeError(

                "ASA 6 encounters remain after final cohort exclusion."

            )



    emergency_column = first_existing_column(

        cohort,

        ["emop", "emergency", "emergency_status", "urgency"],

    )

    if emergency_column is not None:

        emergency_flag, emergency_valid = normalize_emergency(

            cohort[emergency_column]

        )

        rows.append(

            [

                "Emergency surgery, n (%)",

                binary_summary(emergency_flag, emergency_valid),

                binary_summary(

                    emergency_flag.loc[survivors.index],

                    emergency_valid.loc[survivors.index],

                ),

                binary_summary(

                    emergency_flag.loc[deaths.index],

                    emergency_valid.loc[deaths.index],

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

# MAIN

# =============================================================================



def main() -> None:

    reconstructed = reconstruct_current_run()

    table1 = build_table1(reconstructed)



    cohort = reconstructed["cohort"]

    target = reconstructed["target"]

    split_audit = reconstructed["split_audit"]

    cohort_flow = reconstructed["cohort_flow"]



    print("=" * 100)

    print("TABLE 1 — FINAL TEMPORAL-SAFE STRICT-OOF MORTALITY COHORT")

    print("=" * 100)

    print(f"Run directory: {RUN_DIR}")

    print(f"Source dataset: {reconstructed['data_path']}")

    print(f"Target: {reconstructed['target_column']}")

    print(f"Corrected surgical encounters: {len(cohort):,}")

    print(f"Deaths: {int(target.sum()):,}")

    print(f"30-day mortality prevalence: {100.0 * target.mean():.3f}%")

    print("Final cohort and split audit: PASSED")



    print("\nCohort flow:")

    print(cohort_flow.to_string(index=False))



    print("\nSplit audit:")

    print(split_audit.to_string(index=False))



    print("\nTable 1:")

    print(table1.to_string(index=False))



    table1.to_csv(TABLE1_PATH, index=False)

    cohort_flow.to_csv(COHORT_FLOW_PATH, index=False)

    split_audit.to_csv(SPLIT_AUDIT_PATH, index=False)



    print("\nSaved:")

    print(f"  {TABLE1_PATH}")

    print(f"  {COHORT_FLOW_PATH}")

    print(f"  {SPLIT_AUDIT_PATH}")



    try:

        display(table1)  # type: ignore[name-defined]

    except NameError:

        pass





if __name__ == "__main__":

    main()