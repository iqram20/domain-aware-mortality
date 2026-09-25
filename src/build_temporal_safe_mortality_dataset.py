#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/athena/madelab/scratch/iqh4001/VitalDB/Inspire")

FINAL = ROOT / "Results/Final_Datasets/df_final_mortality_2026-05-24_12-44.csv"
OPS   = ROOT / "Data/operations.csv"
DIAG  = ROOT / "Data/diagnosis.csv"

OUT = (
    ROOT
    / "Results/Final_Datasets/"
    / "df_final_mortality_2026-05-24_12-44_temporal_safe.csv"
)

AUDIT = (
    ROOT
    / "Results/Final_Datasets/"
    / "temporal_safe_diagnosis_cci_audit.csv"
)

SUMMARY = (
    ROOT
    / "Results/Final_Datasets/"
    / "temporal_safe_diagnosis_cci_summary.json"
)


# ============================================================
# Same diagnosis chapter grouping used by final dataset
# ============================================================

ICD10_CHAPTER_MAP = {
    "A": "Infectious",
    "B": "Infectious",
    "C": "Neoplasms",
    "D": "Neoplasms",
    "E": "Endocrine",
    "F": "Mental",
    "G": "Nervous",
    "H": "EyeEar",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
}


# ============================================================
# Exact existing Charlson implementation
# ============================================================

CCI_WEIGHTS = {
    "MI": 1,
    "CHF": 1,
    "PVD": 1,
    "Stroke": 1,
    "Dementia": 1,
    "COPD": 1,
    "ConnectiveTissue": 1,
    "Ulcer": 1,
    "MildLiver": 1,
    "DM": 1,
    "DM_Comp": 2,
    "Hemiplegia": 2,
    "Renal": 2,
    "Cancer": 2,
    "Leukemia": 2,
    "Lymphoma": 2,
    "ModerateLiver": 3,
    "Metastatic": 6,
    "HIV": 6,
}

CCI_ICD10_MAP = {
    "I21": "MI",
    "I22": "MI",
    "I252": "MI",

    "I50": "CHF",
    "I11.0": "CHF",
    "I13.0": "CHF",
    "I13.2": "CHF",

    "I70": "PVD",
    "I71": "PVD",
    "I73.1": "PVD",
    "I73.9": "PVD",
    "I77.1": "PVD",

    "I60": "Stroke",
    "I61": "Stroke",
    "I62": "Stroke",
    "I63": "Stroke",
    "I64": "Stroke",

    "F00": "Dementia",
    "F01": "Dementia",
    "F02": "Dementia",
    "F03": "Dementia",
    "G30": "Dementia",

    "J40": "COPD",
    "J41": "COPD",
    "J42": "COPD",
    "J43": "COPD",
    "J44": "COPD",
    "J45": "COPD",

    "M05": "ConnectiveTissue",
    "M06": "ConnectiveTissue",
    "M32": "ConnectiveTissue",
    "M33": "ConnectiveTissue",
    "M34": "ConnectiveTissue",
    "M35": "ConnectiveTissue",
    "M36": "ConnectiveTissue",

    "K25": "Ulcer",
    "K26": "Ulcer",
    "K27": "Ulcer",
    "K28": "Ulcer",

    "K70": "MildLiver",
    "K71": "MildLiver",
    "K73": "MildLiver",
    "K74.0": "MildLiver",
    "K74.2": "MildLiver",
    "K74.6": "MildLiver",

    "K72": "ModerateLiver",
    "K76.6": "ModerateLiver",
    "K76.7": "ModerateLiver",

    "E10.9": "DM",
    "E11.9": "DM",
    "E13.9": "DM",

    "E10.2": "DM_Comp",
    "E11.2": "DM_Comp",
    "E13.2": "DM_Comp",
    "E14.2": "DM_Comp",

    "G81": "Hemiplegia",
    "G82": "Hemiplegia",

    "N18": "Renal",
    "N19": "Renal",
    "N25": "Renal",
    "Z99.2": "Renal",

    "C00": "Cancer",
    "C01": "Cancer",
    "C02": "Cancer",
    "C03": "Cancer",
    "C04": "Cancer",
    "C05": "Cancer",
    "C06": "Cancer",
    "C07": "Cancer",
    "C08": "Cancer",
    "C09": "Cancer",
    "C10": "Cancer",
    "C11": "Cancer",
    "C12": "Cancer",
    "C13": "Cancer",
    "C14": "Cancer",
    "C15": "Cancer",
    "C16": "Cancer",
    "C17": "Cancer",
    "C18": "Cancer",
    "C19": "Cancer",
    "C20": "Cancer",
    "C21": "Cancer",
    "C22": "Cancer",
    "C23": "Cancer",
    "C24": "Cancer",

    "C91": "Leukemia",
    "C92": "Leukemia",
    "C93": "Leukemia",
    "C94": "Leukemia",
    "C95": "Leukemia",

    "C81": "Lymphoma",
    "C82": "Lymphoma",
    "C83": "Lymphoma",
    "C84": "Lymphoma",
    "C85": "Lymphoma",
    "C88": "Lymphoma",

    "C77": "Metastatic",
    "C78": "Metastatic",
    "C79": "Metastatic",
    "C80": "Metastatic",

    "B20": "HIV",
    "B21": "HIV",
    "B22": "HIV",
    "B24": "HIV",
}


def map_icd10_to_cci_category(code):
    if pd.isna(code):
        return None

    code = str(code).upper().replace(".", "")

    for prefix, category in CCI_ICD10_MAP.items():
        if code.startswith(prefix.replace(".", "")):
            return category

    return None


def compute_charlson_safe(df_diag):
    df = df_diag[["subject_id", "icd10_cm"]].dropna().copy()

    df["CCI_Category"] = (
        df["icd10_cm"]
        .map(map_icd10_to_cci_category)
    )

    df = df[df["CCI_Category"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["subject_id"])

    flags = (
        pd.crosstab(
            df["subject_id"],
            df["CCI_Category"],
        )
        .clip(upper=1)
    )

    flags.columns = [
        f"CCI_{c}"
        for c in flags.columns
    ]

    flags = flags.reset_index()

    flags["CCI_score"] = 0

    for category, weight in CCI_WEIGHTS.items():
        col = f"CCI_{category}"

        if col in flags.columns:
            flags["CCI_score"] += (
                flags[col] * weight
            )

    def burden(score):
        if pd.isna(score):
            return np.nan
        if score == 0:
            return "Very Low"
        if score <= 2:
            return "Mild"
        if score <= 4:
            return "Moderate"
        return "Severe"

    flags["CCI_burden_class"] = (
        flags["CCI_score"].map(burden)
    )

    return flags


# ============================================================
# Load
# ============================================================

print("Loading datasets...")

original = pd.read_csv(
    FINAL,
    low_memory=False,
)

ops = pd.read_csv(
    OPS,
    low_memory=False,
)

diag = pd.read_csv(
    DIAG,
    low_memory=False,
)

corrected = original.copy()


# ============================================================
# Verify frozen cohort
# ============================================================

assert original["subject_id"].is_unique

print("Rows:", f"{len(original):,}")
print("Unique subjects:", f"{original.subject_id.nunique():,}")
print("Unique op_id:", f"{original.op_id.nunique():,}")


# ============================================================
# Recover prediction cutoff for each frozen operation
# Keep times numeric — INSPIRE uses minute offsets.
# ============================================================

timing_cols = [
    "subject_id",
    "op_id",
    "opend_time",
    "anend_time",
    "orout_time",
]

index_ops = (
    original[["subject_id", "op_id"]]
    .merge(
        ops[timing_cols],
        on=["subject_id", "op_id"],
        how="left",
        validate="one_to_one",
    )
)

for col in [
    "opend_time",
    "anend_time",
    "orout_time",
]:
    index_ops[col] = pd.to_numeric(
        index_ops[col],
        errors="coerce",
    )

# Model contains surgical/anesthesia/OR-duration and intraoperative
# information, so prediction cutoff is the end of the perioperative
# encounter: latest available end time.
index_ops["prediction_cutoff"] = (
    index_ops[
        [
            "opend_time",
            "anend_time",
            "orout_time",
        ]
    ]
    .max(axis=1, skipna=True)
)

print(
    "Missing prediction cutoff:",
    f"{index_ops.prediction_cutoff.isna().sum():,}",
)


# ============================================================
# Temporally filter diagnosis table
# ============================================================

diag["chart_time"] = pd.to_numeric(
    diag["chart_time"],
    errors="coerce",
)

d = diag.merge(
    index_ops[
        [
            "subject_id",
            "prediction_cutoff",
        ]
    ],
    on="subject_id",
    how="inner",
)

linked_records = len(d)

valid_timing = (
    d["chart_time"].notna()
    & d["prediction_cutoff"].notna()
)

safe_mask = (
    valid_timing
    & (
        d["chart_time"]
        <= d["prediction_cutoff"]
    )
)

post_mask = (
    valid_timing
    & (
        d["chart_time"]
        > d["prediction_cutoff"]
    )
)

diag_safe = d.loc[
    safe_mask,
    [
        "subject_id",
        "chart_time",
        "icd10_cm",
    ],
].copy()

print("\nDiagnosis timing:")
print("  Linked records      :", f"{linked_records:,}")
print("  Safe records        :", f"{safe_mask.sum():,}")
print("  Post-cutoff removed :", f"{post_mask.sum():,}")
print(
    "  Removed percentage :",
    f"{100 * post_mask.sum() / max(valid_timing.sum(), 1):.2f}%"
)


# ============================================================
# Rebuild DIAG_* as TRUE BINARY chapter indicators
# ============================================================

diag_safe["chapter"] = (
    diag_safe["icd10_cm"]
    .astype(str)
    .str[0]
    .map(ICD10_CHAPTER_MAP)
)

diag_chapter = diag_safe[
    diag_safe["chapter"].notna()
].copy()

safe_diag_flags = (
    pd.crosstab(
        diag_chapter["subject_id"],
        diag_chapter["chapter"],
    )
    .clip(upper=1)
)

safe_diag_flags.columns = [
    f"DIAG_{c}"
    for c in safe_diag_flags.columns
]

safe_diag_flags = (
    safe_diag_flags
    .reset_index()
)


# ============================================================
# Rebuild CCI from same temporally safe diagnoses
# ============================================================

safe_cci = compute_charlson_safe(
    diag_safe
)


# ============================================================
# Replace ONLY existing DIAG_* and CCI_* columns
# ============================================================

diag_cols = [
    c for c in original.columns
    if c.startswith("DIAG_")
]

cci_cols = [
    c for c in original.columns
    if c.startswith("CCI_")
]

print("\nExisting frozen diagnosis columns:", len(diag_cols))
print(diag_cols)

print("\nExisting frozen CCI columns:", len(cci_cols))
print(cci_cols)


# Prepare row-aligned replacements
keys = original[["subject_id"]].copy()

diag_aligned = keys.merge(
    safe_diag_flags,
    on="subject_id",
    how="left",
)

cci_aligned = keys.merge(
    safe_cci,
    on="subject_id",
    how="left",
)


# For existing diagnosis columns:
# - subject with at least one eligible chapter gets 0/1
# - subject with no eligible pre/periop chapter remains missing,
#   matching original left-merge behavior.
subjects_with_safe_diag = set(
    safe_diag_flags["subject_id"]
)

has_safe_diag = (
    original["subject_id"]
    .isin(subjects_with_safe_diag)
    .to_numpy()
)

for col in diag_cols:

    if col in diag_aligned.columns:
        values = pd.to_numeric(
            diag_aligned[col],
            errors="coerce",
        )

        values = values.where(
            ~has_safe_diag,
            values.fillna(0),
        )

        corrected[col] = values

    else:
        corrected[col] = np.where(
            has_safe_diag,
            0.0,
            np.nan,
        )


# Existing CCI columns only
for col in cci_cols:

    if col in cci_aligned.columns:
        corrected[col] = cci_aligned[col]

    else:
        # Preserve original merge semantics:
        # absent CCI category = 0 only for subjects
        # represented in safe CCI table.
        has_cci = (
            original["subject_id"]
            .isin(safe_cci["subject_id"])
            .to_numpy()
        )

        corrected[col] = np.where(
            has_cci,
            0.0,
            np.nan,
        )


# ============================================================
# Critical safety checks
# ============================================================

assert len(corrected) == len(original)

assert corrected["subject_id"].equals(
    original["subject_id"]
)

assert corrected["op_id"].equals(
    original["op_id"]
)

assert corrected["mortality_30d"].equals(
    original["mortality_30d"]
)

modified_cols = set(
    diag_cols + cci_cols
)

untouched_cols = [
    c for c in original.columns
    if c not in modified_cols
]

assert corrected[untouched_cols].equals(
    original[untouched_cols]
), "ERROR: A non-DIAG/CCI column changed."


# ============================================================
# Feature-change audit
# ============================================================

rows = []

for col in diag_cols + cci_cols:

    old = original[col]
    new = corrected[col]

    same = (
        old.eq(new)
        | (
            old.isna()
            & new.isna()
        )
    )

    numeric_old = pd.to_numeric(
        old,
        errors="coerce",
    )

    numeric_new = pd.to_numeric(
        new,
        errors="coerce",
    )

    rows.append(
        {
            "feature": col,
            "changed_rows": int((~same).sum()),
            "changed_percent": float(
                100 * (~same).mean()
            ),
            "old_nonmissing": int(old.notna().sum()),
            "new_nonmissing": int(new.notna().sum()),
            "old_positive": int(
                (numeric_old > 0).sum()
            ),
            "new_positive": int(
                (numeric_new > 0).sum()
            ),
            "old_unique": int(
                old.nunique(dropna=True)
            ),
            "new_unique": int(
                new.nunique(dropna=True)
            ),
        }
    )

audit = (
    pd.DataFrame(rows)
    .sort_values(
        "changed_rows",
        ascending=False,
    )
)

AUDIT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.to_csv(
    AUDIT,
    index=False,
)


# ============================================================
# Patient-level change
# ============================================================

if diag_cols + cci_cols:

    comparisons = []

    for col in diag_cols + cci_cols:
        a = original[col]
        b = corrected[col]

        comparisons.append(
            ~(
                a.eq(b)
                | (
                    a.isna()
                    & b.isna()
                )
            )
        )

    change_matrix = pd.concat(
        comparisons,
        axis=1,
    )

    patient_changed = (
        change_matrix.any(axis=1)
    )

else:
    patient_changed = pd.Series(
        False,
        index=original.index,
    )


# ============================================================
# Save corrected dataset
# ============================================================

corrected.to_csv(
    OUT,
    index=False,
)


summary = {
    "source_dataset": str(FINAL),
    "corrected_dataset": str(OUT),
    "rows": int(len(corrected)),
    "unique_subjects": int(
        corrected["subject_id"].nunique()
    ),
    "unique_operations": int(
        corrected["op_id"].nunique()
    ),
    "deaths": int(
        corrected["mortality_30d"].sum()
    ),
    "diagnosis_records_linked": int(
        linked_records
    ),
    "diagnosis_records_safe": int(
        safe_mask.sum()
    ),
    "diagnosis_records_removed_post_cutoff": int(
        post_mask.sum()
    ),
    "diagnosis_records_removed_percent": float(
        100
        * post_mask.sum()
        / max(valid_timing.sum(), 1)
    ),
    "diag_columns_replaced": len(
        diag_cols
    ),
    "cci_columns_replaced": len(
        cci_cols
    ),
    "patients_with_any_diag_or_cci_change": int(
        patient_changed.sum()
    ),
    "patients_with_any_diag_or_cci_change_percent": float(
        100 * patient_changed.mean()
    ),
    "non_diag_cci_columns_changed": 0,
    "prediction_cutoff_definition": (
        "maximum available opend_time, "
        "anend_time, orout_time"
    ),
}

with open(
    SUMMARY,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 100)
print("CORRECTED DATASET COMPLETE")
print("=" * 100)

for k, v in summary.items():
    print(f"{k}: {v}")

print("\nTop changed features:")
print(
    audit.head(30).to_string(
        index=False
    )
)

print("\nSaved:")
print(" ", OUT)
print(" ", AUDIT)
print(" ", SUMMARY)

print("\nPASS: all non-DIAG/CCI columns are unchanged.")
