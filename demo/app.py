"""Interactive Streamlit demo for the 30-day readmission model."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.demo_logic import (  # noqa: E402
    ADMISSION_SOURCES,
    ADMISSION_TYPES,
    DIAGNOSIS_CODES,
    DISCHARGE_DESTINATIONS,
    PRESETS,
    SPECIALTIES,
    build_patient_row,
    observed_risk_neighborhood,
    predict,
    risk_band,
    risk_percentile,
    score_clinical_baselines,
    what_if_deltas,
)


ARTIFACT = ROOT / "demo" / "artifacts" / "readmission_demo.joblib"

st.set_page_config(
    page_title="Reingreso 30 días · Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --ink: #15353d; --muted: #587079; --teal: #087f8c; --teal-dark: #075965;
  --aqua: #e7f5f5; --orange: #e7743b; --sand: #fff6ec; --paper: #fbfdfd;
}
html, body, [class*="css"] { font-family: "Avenir Next", "Segoe UI", sans-serif; color: var(--ink); }
h1, h2, h3 { font-family: "Avenir Next", "Segoe UI", sans-serif !important; font-weight: 750; letter-spacing: -0.025em; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.stApp { background: linear-gradient(180deg, #f4fbfb 0, #ffffff 280px); }
[data-testid="stSidebar"] { background: #f7fbfb; border-right: 1px solid #dceced; }
.hero {
  padding: 1.4rem 1.65rem; border-radius: 22px; color: white;
  background: linear-gradient(120deg, #075965 0%, #087f8c 62%, #1a9ca3 100%);
  box-shadow: 0 16px 45px rgba(7,89,101,.18); margin-bottom: 1rem;
}
.hero h1 { color: white; font-size: 2.15rem; margin: 0 0 .25rem; }
.hero p { color: #def5f5; font-size: 1.02rem; margin: 0; max-width: 800px; }
.academic {
  background: #fff7e9; border: 1px solid #f5d8ad; border-radius: 12px;
  padding: .72rem 1rem; color: #765122; margin-bottom: 1.2rem;
}
.risk-card {
  background: white; border: 1px solid #dcebec; border-radius: 22px;
  padding: 1.25rem; box-shadow: 0 10px 30px rgba(12,84,93,.08);
}
.dial {
  width: 214px; height: 214px; border-radius: 50%; margin: .2rem auto .8rem;
  display: grid; place-items: center; position: relative;
}
.dial:after {
  content: ""; width: 164px; height: 164px; border-radius: 50%; background: white;
  position: absolute; box-shadow: inset 0 0 0 1px #e5efef;
}
.dial-content { position: relative; z-index: 1; text-align: center; }
.dial-number { font-family: "Avenir Next", "Segoe UI", sans-serif; font-weight: 800; font-size: 2.7rem; line-height: 1; }
.dial-label { color: var(--muted); font-size: .82rem; margin-top: .35rem; }
.status {
  border-radius: 999px; font-weight: 700; text-align: center; padding: .5rem .8rem;
  width: fit-content; margin: 0 auto;
}
.status-high { color: #8b321e; background: #ffeadf; }
.status-mid { color: #8a5d0a; background: #fff2c9; }
.status-low { color: #08615f; background: #dcf4ef; }
.mini-card {
  background: white; border: 1px solid #dcebec; border-radius: 16px;
  padding: 1rem 1.1rem; min-height: 118px;
}
.mini-kicker { color: var(--muted); font-size: .8rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.mini-value { color: var(--ink); font: 800 1.75rem "Avenir Next", "Segoe UI", sans-serif; margin: .2rem 0; }
.mini-note { color: var(--muted); font-size: .82rem; line-height: 1.3; }
.flow {
  display: flex; gap: .65rem; align-items: stretch; flex-wrap: wrap; margin: 1rem 0;
}
.flow-card {
  flex: 1; min-width: 145px; padding: 1rem; border-radius: 16px; background: white;
  border: 1px solid #dcebec; text-align: center;
}
.flow-card strong { display: block; font: 800 1.8rem "Avenir Next", "Segoe UI", sans-serif; color: var(--teal-dark); }
.flow-card span { color: var(--muted); font-size: .84rem; }
.flow-arrow { display:grid; place-items:center; color: var(--orange); font-size: 1.5rem; font-weight: 800; }
.callout {
  padding: 1rem 1.1rem; border-left: 4px solid var(--teal); border-radius: 0 12px 12px 0;
  background: var(--aqua); color: #214d54;
}
.orange-callout { border-left-color: var(--orange); background: var(--sand); }
.metric-note { color: var(--muted); font-size: .78rem; margin-top: -.45rem; }
[data-testid="stMetricValue"] { font-family: "Avenir Next", "Segoe UI", sans-serif; }
div[data-testid="stForm"] { border: none; padding: 0; }
@media (max-width: 760px) { .hero h1 { font-size: 1.65rem; } .dial { width: 180px; height: 180px; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    if not ARTIFACT.exists():
        raise FileNotFoundError(
            "Falta el modelo. Ejecutá `.venv/bin/python demo/build_demo_model.py`."
        )
    return joblib.load(ARTIFACT)


bundle = load_bundle()
threshold = float(bundle["threshold_top20"])
prevalence = float(bundle["prevalence"])

st.markdown(
    """
<section class="hero">
  <h1>¿A quién priorizar después del alta?</h1>
  <p>Demo interactiva del modelo de reingreso hospitalario a 30 días.
  Convertí un perfil de alta en una decisión operativa —y explorá por qué cambia.</p>
</section>
<div class="academic"><b>Demo académica, no herramienta clínica.</b>
Datos históricos de EE. UU. (1999–2008), sin validación local ni temporal. No ingresar datos identificatorios.</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Construí un perfil de alta")
    preset_name = st.radio(
        "Caso de ejemplo",
        list(PRESETS),
        index=1,
        horizontal=False,
        help="Elegí un caso y después modificá cualquier dato para ver el riesgo en vivo.",
    )
    preset = PRESETS[preset_name]
    key = preset_name.replace(" ", "_")

    with st.expander("1 · Persona e internación", expanded=True):
        age = st.selectbox(
            "Edad",
            [f"[{i}-{i+10})" for i in range(20, 100, 10)],
            index=[f"[{i}-{i+10})" for i in range(20, 100, 10)].index(preset["age"]),
            key=f"age_{key}",
        )
        gender = st.radio(
            "Sexo registrado",
            ["Female", "Male"],
            index=["Female", "Male"].index(preset["gender"]),
            format_func={"Female": "Femenino", "Male": "Masculino"}.get,
            horizontal=True,
            key=f"gender_{key}",
        )
        admission = st.selectbox(
            "Tipo de admisión",
            list(ADMISSION_TYPES),
            index=list(ADMISSION_TYPES).index(preset["admission"]),
            key=f"admission_{key}",
        )
        source = st.selectbox(
            "Origen",
            list(ADMISSION_SOURCES),
            index=list(ADMISSION_SOURCES).index(preset["source"]),
            key=f"source_{key}",
        )
        discharge = st.selectbox(
            "Destino al alta",
            list(DISCHARGE_DESTINATIONS),
            index=list(DISCHARGE_DESTINATIONS).index(preset["discharge"]),
            key=f"discharge_{key}",
        )
        stay = st.slider(
            "Días de internación",
            1,
            14,
            preset["stay"],
            key=f"stay_{key}",
        )
        specialty = st.selectbox(
            "Especialidad tratante",
            SPECIALTIES,
            index=SPECIALTIES.index(preset["specialty"]),
            key=f"specialty_{key}",
        )

    with st.expander("2 · Utilización previa", expanded=True):
        inpatient = st.slider(
            "Internaciones en el último año",
            0,
            10,
            preset["inpatient"],
            key=f"inpatient_{key}",
            help="La variable más informativa del modelo.",
        )
        emergency = st.slider(
            "Visitas a guardia en el último año",
            0,
            10,
            preset["emergency"],
            key=f"emergency_{key}",
        )
        outpatient = st.slider(
            "Consultas ambulatorias en el último año",
            0,
            20,
            preset["outpatient"],
            key=f"outpatient_{key}",
        )

    with st.expander("3 · Diagnósticos y complejidad", expanded=False):
        diag1 = st.selectbox(
            "Diagnóstico principal",
            list(DIAGNOSIS_CODES),
            index=list(DIAGNOSIS_CODES).index(preset["diag1"]),
            key=f"diag1_{key}",
        )
        diag2 = st.selectbox(
            "Diagnóstico secundario",
            list(DIAGNOSIS_CODES),
            index=list(DIAGNOSIS_CODES).index(preset["diag2"]),
            key=f"diag2_{key}",
        )
        diag3 = st.selectbox(
            "Tercer diagnóstico",
            list(DIAGNOSIS_CODES),
            index=list(DIAGNOSIS_CODES).index(preset["diag3"]),
            key=f"diag3_{key}",
        )
        diagnoses = st.slider(
            "Cantidad de diagnósticos",
            1,
            16,
            preset["diagnoses"],
            key=f"diagnoses_{key}",
        )
        procedures = st.slider(
            "Cantidad de procedimientos",
            0,
            6,
            preset["procedures"],
            key=f"procedures_{key}",
        )

    with st.expander("4 · Laboratorio y tratamiento", expanded=False):
        labs = st.slider(
            "Pruebas de laboratorio",
            1,
            100,
            preset["labs"],
            key=f"labs_{key}",
        )
        medications = st.slider(
            "Medicamentos durante la internación",
            1,
            40,
            preset["medications"],
            key=f"medications_{key}",
        )
        a1c = st.selectbox(
            "Resultado HbA1c",
            ["None", "Norm", ">7", ">8"],
            index=["None", "Norm", ">7", ">8"].index(preset["a1c"]),
            format_func=lambda value: {
                "None": "No medida",
                "Norm": "Normal",
                ">7": "Entre 7 y 8 %",
                ">8": "Mayor a 8 %",
            }[value],
            key=f"a1c_{key}",
        )
        glucose = st.selectbox(
            "Glucemia sérica máxima",
            ["None", "Norm", ">200", ">300"],
            index=["None", "Norm", ">200", ">300"].index(preset["glucose"]),
            format_func=lambda value: "No medida" if value == "None" else value,
            key=f"glucose_{key}",
        )
        insulin = st.selectbox(
            "Insulina",
            ["No", "Steady", "Up", "Down"],
            index=["No", "Steady", "Up", "Down"].index(preset["insulin"]),
            format_func=lambda value: {
                "No": "No indicada",
                "Steady": "Sin cambios",
                "Up": "Aumentada",
                "Down": "Disminuida",
            }[value],
            key=f"insulin_{key}",
        )
        metformin = st.selectbox(
            "Metformina",
            ["No", "Steady", "Up", "Down"],
            index=["No", "Steady", "Up", "Down"].index(preset["metformin"]),
            format_func=lambda value: {
                "No": "No indicada",
                "Steady": "Sin cambios",
                "Up": "Aumentada",
                "Down": "Disminuida",
            }[value],
            key=f"metformin_{key}",
        )
        change = st.radio(
            "¿Cambió la medicación antidiabética?",
            ["No", "Ch"],
            index=["No", "Ch"].index(preset["change"]),
            format_func={"No": "No", "Ch": "Sí"}.get,
            horizontal=True,
            key=f"change_{key}",
        )
        diabetes_med = st.radio(
            "¿Indicación antidiabética al alta?",
            ["Yes", "No"],
            index=["Yes", "No"].index(preset["diabetes_med"]),
            format_func={"Yes": "Sí", "No": "No"}.get,
            horizontal=True,
            key=f"diabetes_med_{key}",
        )

values = {
    "age": age,
    "gender": gender,
    "admission": admission,
    "source": source,
    "discharge": discharge,
    "stay": stay,
    "specialty": specialty,
    "labs": labs,
    "procedures": procedures,
    "medications": medications,
    "diagnoses": diagnoses,
    "outpatient": outpatient,
    "emergency": emergency,
    "inpatient": inpatient,
    "diag1": diag1,
    "diag2": diag2,
    "diag3": diag3,
    "a1c": a1c,
    "glucose": glucose,
    "insulin": insulin,
    "metformin": metformin,
    "change": change,
    "diabetes_med": diabetes_med,
}

patient = build_patient_row(bundle["defaults"], values)
risk = predict(bundle, patient)
percentile = risk_percentile(bundle["test_probabilities"], risk)
band = risk_band(risk, threshold, prevalence)
neighborhood = observed_risk_neighborhood(bundle, risk)
clinical = score_clinical_baselines(bundle, values)
deltas = what_if_deltas(bundle, patient)
status_class = "status-high" if risk >= threshold else ("status-mid" if risk >= prevalence else "status-low")
color = "#e7743b" if risk >= threshold else ("#d8a323" if risk >= prevalence else "#138b83")

tab_patient, tab_impact, tab_model = st.tabs(
    ["Evaluar un alta", "Impacto operativo", "Cómo funciona"]
)

with tab_patient:
    left, right = st.columns([0.82, 1.35], gap="large")
    with left:
        degrees = min(risk / 0.40, 1.0) * 360
        st.markdown(
            f"""
<div class="risk-card">
  <div class="dial" style="background:conic-gradient({color} 0deg {degrees:.1f}deg, #e8f0f0 {degrees:.1f}deg 360deg)">
    <div class="dial-content">
      <div class="dial-number">{risk:.1%}</div>
      <div class="dial-label">riesgo estimado a 30 días</div>
    </div>
  </div>
  <div class="status {status_class}">{band}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(
            "La escala visual llega a 40 % para hacer visibles las diferencias. "
            "El valor central es la probabilidad recalibrada."
        )

    with right:
        st.subheader("De probabilidad a decisión")
        a, b, c = st.columns(3)
        with a:
            st.markdown(
                f"""<div class="mini-card"><div class="mini-kicker">Posición en la cohorte</div>
                <div class="mini-value">P{percentile * 100:.0f}</div>
                <div class="mini-note">Mayor riesgo estimado que el {percentile:.0%} de las altas de test.</div></div>""",
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                f"""<div class="mini-card"><div class="mini-kicker">Cupo operativo</div>
                <div class="mini-value">{threshold:.1%}</div>
                <div class="mini-note">Umbral que seleccionó el 20 % de mayor riesgo en validación.</div></div>""",
                unsafe_allow_html=True,
            )
        with c:
            st.markdown(
                f"""<div class="mini-card"><div class="mini-kicker">Pacientes similares</div>
                <div class="mini-value">{neighborhood['observed_rate']:.1%}</div>
                <div class="mini-note">Reingreso observado en el decil de riesgo {int(neighborhood['decile'])} del test.</div></div>""",
                unsafe_allow_html=True,
            )

        if risk >= threshold:
            st.markdown(
                """<div class="callout orange-callout"><b>Acción sugerida por el caso de uso:</b>
                incluir esta alta en el cupo de seguimiento: llamado a 48–72 h, conciliación de medicación
                y turno temprano. La app prioriza; no indica que el reingreso sea inevitable.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="callout"><b>No entra en el 20 % priorizado.</b>
                Esto no significa “sin riesgo”: el modelo deja fuera aproximadamente 6 de cada 10
                reingresos porque el cupo es limitado.</div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    compare_left, explain_right = st.columns([0.9, 1.2], gap="large")
    with compare_left:
        st.subheader("Modelo vs. referencias simples")
        comparison = pd.DataFrame(
            {
                "Riesgo": [
                    risk,
                    clinical["LACE"],
                    clinical["HOSPITAL"],
                    prevalence,
                    threshold,
                ]
            },
            index=[
                "Modelo v2",
                f"LACE aprox. ({clinical['LACE_points']} pt)",
                f"HOSPITAL parcial ({clinical['HOSPITAL_points']} pt)",
                "Promedio cohorte",
                "Umbral 20 %",
            ],
        )
        st.bar_chart(comparison, color="#087f8c", height=320)
        st.caption(
            "LACE y HOSPITAL son aproximados: UCI no tiene hemoglobina ni sodio "
            "y solo incluye tres diagnósticos."
        )

    with explain_right:
        st.subheader("¿Qué está moviendo este resultado?")
        display_deltas = (
            deltas.head(7)
            .assign(
                **{
                    "Cambio en riesgo (p.p.)": lambda frame: frame.delta * 100,
                    "Dirección": lambda frame: frame.delta.map(
                        lambda delta: "Aumenta" if delta > 0 else "Disminuye"
                    ),
                }
            )
            .set_index("factor")[["Cambio en riesgo (p.p.)"]]
            .sort_values("Cambio en riesgo (p.p.)")
        )
        st.bar_chart(display_deltas, horizontal=True, color="#e7743b", height=310)
        largest = deltas.iloc[0]
        direction = "más alto" if largest.delta > 0 else "más bajo"
        st.markdown(
            f"""<div class="callout"><b>Lectura rápida:</b> el mayor cambio local es
            <b>{largest.factor.lower()}</b>. Con ese grupo de datos en el valor de referencia,
            el riesgo sería {largest.risk_if_reference:.1%}; el perfil actual lo deja
            {abs(largest.delta) * 100:.1f} puntos porcentuales {direction}.</div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            "Análisis “qué pasaría si” de a un grupo de variables por vez. "
            "No es causal, no es una recomendación terapéutica y puede verse afectado por correlaciones."
        )

    with st.expander("Ver el perfil completo enviado al modelo (45 variables)"):
        readable = patient.T.reset_index()
        readable.columns = ["Variable", "Valor"]
        readable["Valor"] = readable["Valor"].astype(str)
        st.dataframe(readable, hide_index=True, width="stretch", height=360)

with tab_impact:
    st.header("¿Qué significa usar un cupo del 20 %?")
    st.write(
        "Proyección por cada **1.000 altas**, usando el desempeño observado en test. "
        "No estima el efecto de la intervención; muestra a quién identificaría el modelo."
    )
    total = 1000
    selected = round(total * 0.20)
    total_events = round(total * bundle["prevalence"])
    captured = round(total_events * bundle["operating_point"]["recall"])
    false_alerts = selected - captured
    missed = total_events - captured
    st.markdown(
        f"""
<div class="flow">
  <div class="flow-card"><strong>{total}</strong><span>altas evaluadas</span></div>
  <div class="flow-arrow">→</div>
  <div class="flow-card"><strong>{selected}</strong><span>reciben seguimiento</span></div>
  <div class="flow-arrow">→</div>
  <div class="flow-card"><strong>{captured}</strong><span>reingresos captados</span></div>
  <div class="flow-card"><strong>{false_alerts}</strong><span>seguidas sin reingreso</span></div>
  <div class="flow-card"><strong>{missed}</strong><span>reingresos fuera del cupo</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="callout orange-callout"><b>El mensaje operativo:</b>
        el modelo concentra aproximadamente <b>{bundle['operating_point']['recall']:.0%}</b>
        de los reingresos en el 20 % de las altas. Pero la precisión es
        <b>{bundle['operating_point']['precision']:.0%}</b>: cerca de 7 de cada 9 personas
        seguidas no reingresan. Tiene sentido para un llamado barato, no automáticamente
        para una visita domiciliaria.</div>""",
        unsafe_allow_html=True,
    )

    st.subheader("El costo de la intervención cambia la decisión")
    scenario = pd.DataFrame(
        [
            {
                "Intervención": "Llamado estructurado",
                "Costo relativo": "1 % de un reingreso",
                "Umbral racional": "5 %",
                "Altas a seguir": "≈91 %",
                "Lectura": "Tan barato que conviene ofrecerlo casi universalmente.",
            },
            {
                "Intervención": "Cupo de la PoC",
                "Costo relativo": "≈3 % de un reingreso",
                "Umbral racional": "15,4 %",
                "Altas a seguir": "20 %",
                "Lectura": "Coherente con llamado + conciliación + turno temprano.",
            },
            {
                "Intervención": "Visita domiciliaria",
                "Costo relativo": "5 % de un reingreso",
                "Umbral racional": "25 %",
                "Altas a seguir": "≈4 %",
                "Lectura": "El cupo debería achicarse y enfocarse en riesgo extremo.",
            },
        ]
    )
    st.dataframe(scenario, hide_index=True, width="stretch")
    st.caption(
        "Supuesto ilustrativo: la intervención evita el 20 % de los reingresos entre quienes la reciben. "
        "Los costos y la eficacia deben medirse localmente."
    )

with tab_model:
    st.header("Qué aprendimos del modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUROC", f"{bundle['metrics']['AUROC']:.3f}")
    m2.metric("Average Precision", f"{bundle['metrics']['AP']:.3f}")
    m3.metric("Lift @20 %", f"{bundle['metrics']['Lift@20']:.2f}×")
    m4.metric("Brier", f"{bundle['metrics']['Brier']:.3f}")
    st.caption(
        "Test por paciente; evidencia secundaria porque el mismo test ya se había consultado en la v1."
    )

    story_a, story_b = st.columns(2, gap="large")
    with story_a:
        st.subheader("La complejidad ayudó, pero poco")
        st.markdown(
            """
- El mejor modelo fue **LightGBM de baja capacidad**.
- Superó a la logística en los **15 folds** de validación cruzada.
- En test, la mejora fue pequeña: AP 0,224 vs. 0,217.
- En el cupo del 20 %, ambos tienen una precisión cercana al 22 %.
"""
        )
    with story_b:
        st.subheader("Desbalance: más datos sintéticos no fue mejor")
        imbalance = pd.DataFrame(
            {
                "AP media": [0.226, 0.225, 0.221, 0.202],
            },
            index=["Sin ajuste", "Pesos", "Undersampling", "SMOTENC"],
        )
        st.bar_chart(imbalance, color="#087f8c", height=245)
        st.caption(
            "Con miles de eventos por fold, el límite era la señal disponible, no la cantidad de positivos."
        )

    st.subheader("Lo que esta app no puede afirmar")
    st.markdown(
        """
1. **No predice causalidad ni evitabilidad.** Un riesgo alto no indica qué intervención funcionará.
2. **No está validada en Argentina.** Usa datos de 130 hospitales de EE. UU. entre 1999 y 2008, con CIE-9.
3. **El outcome es incompleto.** Un “no reingreso” puede ser atención fuera de la red.
4. **Algunas variables llegan tarde.** Diagnósticos y codificación pueden completarse después del alta.
5. **No reemplaza criterio clínico.** Está diseñada para discutir priorización de recursos en clase.
"""
    )
    st.markdown(
        f"""<div class="callout"><b>Modelo servido:</b> {bundle['model_name']} ·
        recalibración Platt · {len(bundle['defaults'])} variables ·
        scikit-learn {bundle['versions']['scikit-learn']} ·
        LightGBM {bundle['versions']['lightgbm']}.</div>""",
        unsafe_allow_html=True,
    )

st.caption(
    "PoC de reingreso hospitalario a 30 días · Lucas Didone · MIA, FIUBA · "
    "Fuente: UCI Diabetes 130-US Hospitals (CC BY 4.0)."
)
