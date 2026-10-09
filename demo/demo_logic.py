"""Pure helpers used by the Streamlit readmission demo."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

import reingresos_utils as ru


DIAGNOSIS_CODES = {
    "Circulatorio": ("428", "Circulatory"),
    "Respiratorio": ("486", "Respiratory"),
    "Digestivo": ("530", "Digestive"),
    "Diabetes": ("250.8", "Diabetes"),
    "Lesiones": ("820", "Injury"),
    "Musculoesquelético": ("715", "Musculoskeletal"),
    "Genitourinario": ("585", "Genitourinary"),
    "Neoplasias": ("174", "Neoplasms"),
    "Otro": ("780", "Other"),
}

ADMISSION_TYPES = {
    "Emergencia": "1",
    "Urgente": "2",
    "Electiva": "3",
    "Trauma": "7",
    "No disponible": "5",
}

ADMISSION_SOURCES = {
    "Guardia": "7",
    "Derivación médica": "1",
    "Transferencia de otro hospital": "4",
    "Transferencia de otra institución": "6",
    "Otro / desconocido": "9",
}

DISCHARGE_DESTINATIONS = {
    "Domicilio": "1",
    "Domicilio con cuidados": "6",
    "Institución de cuidados especializados": "3",
    "Rehabilitación": "22",
    "Otro destino": "18",
}

SPECIALTIES = [
    "InternalMedicine",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Cardiology",
    "Surgery-General",
    "Nephrology",
    "Orthopedics",
    "Other",
]

PRESETS = {
    "Bajo riesgo": {
        "age": "[50-60)",
        "gender": "Female",
        "admission": "Electiva",
        "source": "Derivación médica",
        "discharge": "Domicilio",
        "stay": 2,
        "specialty": "Family/GeneralPractice",
        "labs": 35,
        "procedures": 1,
        "medications": 10,
        "diagnoses": 5,
        "outpatient": 1,
        "emergency": 0,
        "inpatient": 0,
        "diag1": "Musculoesquelético",
        "diag2": "Diabetes",
        "diag3": "Otro",
        "a1c": "Norm",
        "glucose": "None",
        "insulin": "No",
        "metformin": "Steady",
        "change": "No",
        "diabetes_med": "Yes",
    },
    "Riesgo intermedio": {
        "age": "[70-80)",
        "gender": "Female",
        "admission": "Emergencia",
        "source": "Guardia",
        "discharge": "Domicilio",
        "stay": 5,
        "specialty": "InternalMedicine",
        "labs": 47,
        "procedures": 1,
        "medications": 16,
        "diagnoses": 9,
        "outpatient": 0,
        "emergency": 0,
        "inpatient": 0,
        "diag1": "Circulatorio",
        "diag2": "Diabetes",
        "diag3": "Circulatorio",
        "a1c": "None",
        "glucose": "None",
        "insulin": "Steady",
        "metformin": "No",
        "change": "No",
        "diabetes_med": "Yes",
    },
    "Alto riesgo": {
        "age": "[80-90)",
        "gender": "Male",
        "admission": "Emergencia",
        "source": "Guardia",
        "discharge": "Institución de cuidados especializados",
        "stay": 11,
        "specialty": "InternalMedicine",
        "labs": 72,
        "procedures": 3,
        "medications": 24,
        "diagnoses": 14,
        "outpatient": 2,
        "emergency": 3,
        "inpatient": 4,
        "diag1": "Circulatorio",
        "diag2": "Genitourinario",
        "diag3": "Diabetes",
        "a1c": "None",
        "glucose": ">200",
        "insulin": "Up",
        "metformin": "No",
        "change": "Ch",
        "diabetes_med": "Yes",
    },
}


def build_patient_row(defaults: Mapping, values: Mapping) -> pd.DataFrame:
    """Build all 45 model features from the compact demo form."""
    row = dict(defaults)
    row.update(
        {
            "gender": values["gender"],
            "age": values["age"],
            "admission_type_id": ADMISSION_TYPES[values["admission"]],
            "admission_source_id": ADMISSION_SOURCES[values["source"]],
            "discharge_disposition_id": DISCHARGE_DESTINATIONS[values["discharge"]],
            "time_in_hospital": values["stay"],
            "medical_specialty": values["specialty"],
            "num_lab_procedures": values["labs"],
            "num_procedures": values["procedures"],
            "num_medications": values["medications"],
            "number_diagnoses": values["diagnoses"],
            "number_outpatient": values["outpatient"],
            "number_emergency": values["emergency"],
            "number_inpatient": values["inpatient"],
            "A1Cresult": values["a1c"],
            "max_glu_serum": values["glucose"],
            "insulin": values["insulin"],
            "metformin": values["metformin"],
            "change": values["change"],
            "diabetesMed": values["diabetes_med"],
        }
    )
    for index in (1, 2, 3):
        code, group = DIAGNOSIS_CODES[values[f"diag{index}"]]
        row[f"diag_{index}"] = code
        row[f"diag_{index}_group"] = group
    return pd.DataFrame([row], columns=list(defaults))


def predict(bundle: Mapping, row: pd.DataFrame) -> float:
    return float(bundle["model"].predict_proba(row)[:, 1][0])


def risk_percentile(test_probabilities, risk: float) -> float:
    """Percentage of test discharges with risk no greater than this profile."""
    return float(np.mean(np.asarray(test_probabilities) <= risk))


def risk_band(risk: float, threshold: float, prevalence: float) -> str:
    if risk >= threshold:
        return "Prioridad de seguimiento"
    if risk >= prevalence:
        return "Riesgo superior al promedio"
    return "Riesgo inferior al promedio"


def score_clinical_baselines(bundle: Mapping, values: Mapping) -> dict[str, float]:
    """Approximate UCI LACE and HOSPITAL probabilities for the demo profile."""
    diag_codes = [DIAGNOSIS_CODES[values[f"diag{i}"]][0] for i in (1, 2, 3)]
    flags = ru.comorbidity_flags(pd.Series(diag_codes), version=9).any(axis=0).to_frame().T
    charlson = int(ru.charlson_score(flags).iloc[0])
    emergent = values["admission"] in {"Emergencia", "Trauma"} or values["source"] == "Guardia"
    lace = int(ru.lace_score([values["stay"]], [emergent], [charlson], [values["emergency"]])[0])
    oncology = any(values[f"diag{i}"] == "Neoplasias" for i in (1, 2, 3))
    hospital = int(
        ru.hospital_score(
            [np.nan],
            [oncology],
            [np.nan],
            [values["procedures"] > 0],
            [values["admission"] != "Electiva"],
            [values["inpatient"]],
            [values["stay"]],
        )[0]
    )
    lace_p = float(bundle["score_models"]["LACE"].predict_proba(pd.DataFrame({"LACE": [lace]}))[:, 1][0])
    hospital_p = float(
        bundle["score_models"]["HOSPITAL"].predict_proba(
            pd.DataFrame({"HOSPITAL": [hospital]})
        )[:, 1][0]
    )
    return {"LACE": lace_p, "HOSPITAL": hospital_p, "LACE_points": lace, "HOSPITAL_points": hospital}


WHAT_IF_GROUPS = {
    "Internaciones previas": ["number_inpatient"],
    "Destino al alta": ["discharge_disposition_id"],
    "Diagnósticos": [
        "diag_1",
        "diag_2",
        "diag_3",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
        "number_diagnoses",
    ],
    "Edad": ["age"],
    "Duración de la internación": ["time_in_hospital"],
    "Visitas a guardia": ["number_emergency"],
    "Especialidad": ["medical_specialty"],
    "HbA1c": ["A1Cresult"],
    "Tratamiento de diabetes": ["insulin", "metformin", "change", "diabetesMed"],
}


def what_if_deltas(bundle: Mapping, row: pd.DataFrame) -> pd.DataFrame:
    """One-at-a-time risk differences versus the reference profile.

    These are sensitivity analyses, not causal effects or SHAP values.
    """
    current = predict(bundle, row)
    records = []
    for label, columns in WHAT_IF_GROUPS.items():
        alternative = row.copy()
        for column in columns:
            alternative.loc[0, column] = bundle["defaults"][column]
        without = predict(bundle, alternative)
        records.append(
            {
                "factor": label,
                "delta": current - without,
                "risk_if_reference": without,
            }
        )
    return pd.DataFrame(records).sort_values("delta", key=np.abs, ascending=False)


def observed_risk_neighborhood(bundle: Mapping, risk: float) -> dict:
    """Return observed rate in the test-risk decile containing the profile."""
    table = pd.DataFrame(bundle["risk_deciles"])
    row = table[(risk >= table["risk_min"]) & (risk <= table["risk_max"])]
    if row.empty:
        row = table.iloc[[0 if risk < table.risk_min.min() else -1]]
    return row.iloc[0].to_dict()
