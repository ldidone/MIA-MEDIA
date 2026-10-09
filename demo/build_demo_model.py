"""Build the frozen local artifact used by the Streamlit demo.

Run from the project root:
    .venv/bin/python demo/build_demo_model.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reingresos_utils as ru  # noqa: E402


DATA_ARCHIVE = ROOT / "data" / "diabetes.zip"
RESULTS = ROOT / "output" / "v2" / "results_v2.json"
ARTIFACT = ROOT / "demo" / "artifacts" / "readmission_demo.joblib"


def load_cohort() -> pd.DataFrame:
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        with archive.open("diabetic_data.csv") as source:
            raw = pd.read_csv(
                source,
                keep_default_na=False,
                na_values=["?"],
                low_memory=False,
            )
    cohort = raw.loc[
        ~raw.age.isin(["[0-10)", "[10-20)"])
        & ~raw.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])
        & raw.gender.ne("Unknown/Invalid")
    ].copy()
    return cohort.reset_index(drop=True)


def split_indices(cohort: pd.DataFrame):
    groups = cohort.patient_nbr
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=ru.SEED)
    dev, test = next(first.split(cohort, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=ru.SEED + 1)
    train_local, validation_local = next(
        second.split(cohort.iloc[dev], groups=groups.iloc[dev])
    )
    return dev[train_local], dev[validation_local], test


def feature_frame(cohort: pd.DataFrame, train_idx):
    excluded = ["encounter_id", "patient_nbr", "readmitted", "race", "payer_code"]
    features = cohort.drop(columns=excluded).copy()
    for column in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        features[column] = features[column].astype(str)
    for index in (1, 2, 3):
        features[f"diag_{index}_group"] = cohort[f"diag_{index}"].map(ru.icd9_group)
    missing = features.iloc[train_idx].isna().mean()
    dropped = set(missing[missing > 0.80].index)
    dropped |= {
        column
        for column in features
        if features.iloc[train_idx][column].nunique(dropna=False) <= 1
    }
    features = features.drop(columns=sorted(dropped))
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features if column not in numeric]
    return features, numeric, categorical


def clinical_scores(cohort: pd.DataFrame) -> pd.DataFrame:
    flags = ru.comorbidity_flags(cohort.diag_1, 9)
    for index in (2, 3):
        flags |= ru.comorbidity_flags(cohort[f"diag_{index}"], 9)
    charlson = ru.charlson_score(flags)
    emergent = cohort.admission_type_id.isin([1, 7]) | cohort.admission_source_id.eq(7)
    oncology = pd.concat(
        [
            cohort[f"diag_{index}"].map(ru.icd9_group).eq("Neoplasms")
            for index in (1, 2, 3)
        ],
        axis=1,
    ).any(axis=1)
    return pd.DataFrame(
        {
            "LACE": ru.lace_score(
                cohort.time_in_hospital,
                emergent,
                charlson,
                cohort.number_emergency,
            ),
            "HOSPITAL": ru.hospital_score(
                np.nan,
                oncology,
                np.nan,
                cohort.num_procedures.gt(0),
                cohort.admission_type_id.isin([1, 2, 7]),
                cohort.number_inpatient,
                cohort.time_in_hospital,
            ),
        }
    )


def make_defaults(features: pd.DataFrame, numeric, categorical, train_idx) -> dict:
    train = features.iloc[train_idx]
    defaults = {}
    for column in features:
        if column in numeric:
            defaults[column] = float(train[column].median())
        else:
            mode = train[column].mode(dropna=True)
            defaults[column] = str(mode.iloc[0]) if len(mode) else "Missing"
    return defaults


def main() -> None:
    expected = json.loads(RESULTS.read_text(encoding="utf-8"))
    archive_hash = hashlib.sha256(DATA_ARCHIVE.read_bytes()).hexdigest()
    if archive_hash != expected["sha256_zip"]:
        raise RuntimeError("The UCI archive differs from the notebook dataset.")

    cohort = load_cohort()
    y = cohort.readmitted.eq("<30").astype(int)
    train_idx, validation_idx, test_idx = split_indices(cohort)
    features, numeric, categorical = feature_frame(cohort, train_idx)
    if features.columns.tolist() != expected["features"]:
        raise RuntimeError("Demo feature order differs from the executed v2 notebook.")

    model = ru.make_pipeline(
        "LGBM",
        "none",
        numeric,
        categorical,
        params={"num_leaves": 15, "colsample_bytree": 0.5},
        seed=ru.SEED,
    )
    model.fit(features.iloc[train_idx], y.iloc[train_idx])
    calibrated = ru.fit_calibrator(
        model,
        features.iloc[validation_idx],
        y.iloc[validation_idx],
        "sigmoid",
    )

    scores = clinical_scores(cohort)
    score_models = {
        name: LogisticRegression().fit(scores.iloc[train_idx][[name]], y.iloc[train_idx])
        for name in scores
    }

    test_probabilities = calibrated.predict_proba(features.iloc[test_idx])[:, 1]
    test = pd.DataFrame(
        {"risk": test_probabilities, "outcome": y.iloc[test_idx].to_numpy()}
    )
    test["decile"] = pd.qcut(test.risk, 10, labels=False, duplicates="drop") + 1
    deciles = (
        test.groupby("decile", as_index=False)
        .agg(
            n=("outcome", "size"),
            observed_rate=("outcome", "mean"),
            predicted_mean=("risk", "mean"),
            risk_min=("risk", "min"),
            risk_max=("risk", "max"),
        )
        .to_dict("records")
    )

    defaults = make_defaults(features, numeric, categorical, train_idx)
    bundle = {
        "artifact_version": 1,
        "created": "2026-10-08",
        "model": calibrated,
        "score_models": score_models,
        "defaults": defaults,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "test_probabilities": np.sort(test_probabilities),
        "risk_deciles": deciles,
        "threshold_top20": expected["decision"]["threshold_top20_validation"],
        "prevalence": expected["cohort_prevalence"],
        "metrics": expected["test"]["Modelo v2"],
        "confidence_intervals": expected["ci95"],
        "operating_point": expected["operating_top20"],
        "dataset_sha256": archive_hash,
        "model_name": expected["cv"]["chosen"],
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT, compress=3)

    restored = joblib.load(ARTIFACT)
    restored_probabilities = restored["model"].predict_proba(features.iloc[test_idx])[:, 1]
    if not np.allclose(test_probabilities, restored_probabilities):
        raise RuntimeError("Serialized model predictions changed after reload.")
    print(f"Artifact: {ARTIFACT} ({ARTIFACT.stat().st_size / 1_048_576:.2f} MiB)")
    print(
        f"Test: AUROC={ru.metrics(y.iloc[test_idx], test_probabilities)['AUROC']:.4f}, "
        f"AP={ru.metrics(y.iloc[test_idx], test_probabilities)['AP']:.4f}"
    )


if __name__ == "__main__":
    main()
