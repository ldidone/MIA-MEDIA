# PoC de reingreso hospitalario antes de 30 días

Lucas Didone · Primera entrega · 2 de octubre de 2026

## Resultado

Hay señal predictiva moderada que justifica continuar la investigación académica. En test por paciente, la regresión logística obtiene AUROC 0,672 y Average Precision 0,217 (prevalencia 0,114). El 20% de episodios priorizados contiene 38,8% de los reingresos, con precisión 22,1% y lift 1,94. El modelo se eligió por AP en validación, antes de evaluar test.

## Archivos

- `PoC_Reingresos_Diabetes.ipynb`: notebook ya ejecutado, con código, tablas, gráficos y conclusión.
- `requirements.txt`: dependencias. Entorno de referencia: Python 3.12.14.
- `data/diabetes.zip`: descarga pública original de UCI, incluye CSV de datos y diccionario.
- `output/Informe_PoC_Reingresos.pdf`: informe breve de 4 páginas.
- `output/Primera_Entrega_Reingresos.pptx`: 10 diapositivas editables con notas y fuentes.
- `output/results.json`: métricas completas, cohortes, versiones y SHA256 de la descarga.
- `output/metricas_test.csv`, `intervalos_bootstrap.csv`, `capacidad_seguimiento.csv`: resultados numéricos.
- `output/importancia_validacion.csv`: importancia por permutación en validación.
- `output/split_assignments.csv`: asignación auditable de cada episodio/paciente a train, validación o test.
- `output/*.png`: gráficos derivados de la ejecución.

## Ejecución local

Descomprimir y abrir una terminal en esta carpeta:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab PoC_Reingresos_Diabetes.ipynb
```

En Windows, activar con `.venv\Scripts\activate`. Ejecutar todas las celdas en orden. No requiere GPU, credenciales ni API paga. La instalación necesita Internet; después, el dataset incluido permite ejecutar sin conexión. El notebook independiente también puede descargar los datos si falta el ZIP.

## Diseño

Cohorte de 20 años o más, sin fallecimiento ni hospice y con sexo registrado válido. Se retienen 98.490 episodios de 69.311 pacientes. La edad viene agrupada, por lo que no se puede separar 18–19 años. Particiones por paciente, aproximadamente 60/20/20. Cero pacientes compartidos. El preprocesamiento se ajusta solo en entrenamiento. Se conserva la categoría `None` de estudios no realizados.

Se comparan prevalencia, logística y HistGradientBoosting de scikit-learn. Se incluyen ablaciones de utilización previa y un control de etiquetas permutadas. IC95% por 300 remuestreos de pacientes completos. No hay ajuste intensivo de hiperparámetros. La importancia por permutación usa validación y no permite inferir causalidad.

## Interpretación

El endpoint es reingreso registrado antes de 30 días, de cualquier causa. No significa recaída de diabetes ni evento evitable. El caso de uso propuesto es priorizar revisión y seguimiento **al alta**, por episodio. Los datos son históricos, de EE. UU., con captura potencialmente incompleta del seguimiento. El test es aleatorio por pacientes, no temporal ni externo.

De 3.955 altas seleccionadas, 876 tienen reingreso temprano registrado y 3.079 no. Quedan 1.383 eventos fuera del grupo priorizado. La factibilidad técnica no demuestra eficacia clínica. El test ya consultado no debe utilizarse reiteradamente para ajustar futuras versiones.

## Segunda iteración (v2, 8 de octubre de 2026)

`PoC_Reingresos_Diabetes_v2.ipynb` incorpora la devolución de la primera entrega (`feedback.md`) y el pedido de la segunda iteración. Usa la misma cohorte y la misma partición por paciente que la v1, verificada contra `output/split_assignments.csv`. El código común está en `reingresos_utils.py`.

**Resultado.** El modelo se elige con validación cruzada agrupada por paciente y repetida (5 folds × 3) sobre train + validación. Se cruzaron LR, HGB, XGBoost y LightGBM con cuatro estrategias de desbalance: ninguna, pesos de clase, undersampling y SMOTENC. Gana LightGBM de baja capacidad **sin remuestreo** (AP de CV 0,226 ± 0,005 frente a 0,216 de la logística; mejor en los 15 folds). Ninguna técnica de desbalance mejora la AP, y SMOTENC la empeora en todas las familias. En test (ya consultado en la v1, por lo que es evidencia secundaria): AUROC 0,679 (IC95 0,667–0,692), AP 0,224 (0,209–0,242), lift@20 % 1,95. La mejora sobre la logística v1 es pequeña (ΔAP +0,0005 a +0,0135) y amplia sobre LACE aproximado (ΔAP +0,067 a +0,093). La curva de decisión muestra que el cupo de 20 % (umbral ≈0,15) se justifica para intervenciones que cuesten hasta ≈3 % de un reingreso, si evitan el 20 % de los reingresos de quienes las reciben.

**Qué agrega:** baselines LACE y HOSPITAL; recalibración (Platt frente a isotónica) en validación; curvas de decisión con `dcurves`; subgrupos con IC95 %; análisis de HbA1c; diccionario de datos con sistema de origen, disponibilidad al alta y mapeo a CIE-10 / SNOMED CT / LOINC / ATC; ficha TRIPOD+AI con PROBAST; esbozo de ensayo pragmático; registro de corridas con MLflow.

**Validación temporal.** UCI no tiene fechas, así que la v2 no hace partición temporal. `PoC_Reingresos_MIMIC_IV_Temporal.ipynb` implementa la partición temporal por `anchor_year_group`: train 2008–2013, validación 2014–2016, OOT 2017–2019 y 2020–2022. Incluye validación cruzada temporal de ventana expansiva, LACE y HOSPITAL completos y una tabla de degradación por período. Requiere acceso credencial a MIMIC-IV en PhysioNet: si faltan los datos en `data/mimic-iv/hosp/`, el notebook se detiene y muestra los pasos de acceso. Se probó de punta a punta con datos sintéticos que siguen el esquema de MIMIC-IV, sin resultados reales. Los datos de MIMIC no se versionan.

**Archivos nuevos:**
- `PoC_Reingresos_Diabetes_v2.ipynb`: notebook v2 ya ejecutado (≈20 min en CPU de 8 núcleos).
- `PoC_Reingresos_MIMIC_IV_Temporal.ipynb`: experimento temporal en MIMIC-IV (sin ejecutar: requiere datos).
- `reingresos_utils.py`: métricas, modelos, desbalance, CV, LACE/HOSPITAL, Charlson CIE-9/CIE-10, bootstrap agrupado, recalibración y MLflow.
- `output/v2/`: `results_v2.json`, `cv_resumen.csv`, `cv_folds.csv`, `metricas_test.csv`, `intervalos_bootstrap.csv`, `escenarios_costo.csv`, `subgrupos_test.csv`, `diccionario_datos.csv`, `mapeo_cie9_cie10_grupos.csv`, `ficha_modelo.md` y gráficos.
- `mlruns/`: corridas de MLflow (no versionado). Ver con `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

Entorno de la v2: Python 3.13.5 con las versiones fijadas en `requirements.txt` (Python 3.12 también es compatible).

## Demo interactiva para clase

`demo/app.py` es una aplicación Streamlit local que sirve el LightGBM v2 congelado y recalibrado. Permite construir un perfil de alta, ver su riesgo y posición en la cohorte, compararlo con LACE/HOSPITAL, explorar sensibilidades locales y traducir el cupo del 20 % a impacto operativo.

```bash
source .venv/bin/activate
streamlit run demo/app.py
```

Abrir `http://localhost:8501`. No requiere Internet ni reentrena el modelo. El recorrido sugerido y las instrucciones para reconstruir el artefacto están en `demo/README.md`.

La misma aplicación está desplegada como un [Space público en Hugging Face](https://huggingface.co/spaces/ldidone/reingresos-diabetes-demo). La configuración Docker y el script de actualización están en `deploy/huggingface/`.

## Fuentes y licencia del dataset

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. Licencia CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.

Dataset redistribuido sin modificar dentro de `data/diabetes.zip`. Los filtros y transformaciones están documentados en el notebook. No se implica aval de los autores.

Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. https://doi.org/10.1155/2014/781670.

Todos los resultados del informe y las slides proceden de la ejecución incluida, no de métricas publicadas por terceros.
