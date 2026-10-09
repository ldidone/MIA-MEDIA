
### Ficha del modelo: reingreso <30 días, v2 (TRIPOD+AI)

| Ítem | Contenido |
|---|---|
| Uso previsto | Priorizar seguimiento posalta (por episodio) en adultos con diabetes. Apoyo a la decisión, no diagnóstico. Investigación académica; no validado para uso clínico. |
| Fuente de datos | UCI Diabetes 130-US Hospitals 1999–2008 (Health Facts), SHA-256 `f82ac129da2ddd22…`. 130 hospitales de EE. UU. |
| Participantes | 98,490 internaciones de 69,311 pacientes; ≥20 años; sin fallecimiento/hospice. Prevalencia 11.44%. |
| Outcome | Reingreso registrado <30 días, cualquier causa, dentro de la red de datos. |
| Predictores | 45 variables al alta (sin raza ni financiador). Ver `diccionario_datos.csv`. |
| Tamaño muestral | Train 58,870 (6,758 eventos); validación 19,849; test 19,771. |
| Faltantes | Mediana (numéricas) y categoría "Missing" (categóricas), ajustadas en cada fold de train. |
| Desarrollo | Grilla LR/HGB/XGBoost/LightGBM × desbalance; CV 5×3 agrupada por paciente; regla de 1 EE. Elegido: **LGBM-num_leaves15-colsample_bytree0.5 \| none**. Recalibración sigmoid en validación. |
| Desempeño (test, secundario) | AUROC 0.679 (IC95 0.667–0.692); AP 0.224 (0.209–0.241); Brier 0.0963; lift@20 % 1.95. |
| Comparadores | LACE aprox. AP 0.145; HOSPITAL parcial AP 0.157; v1 LR AP 0.217. |
| Punto operativo | Cupo 20 % ≡ umbral 0.154; coherente con intervenciones de costo ≤ 3.1 % de un reingreso (si evitan el 20%). |
| Equidad | Métricas por sexo, edad, HbA1c y destino con IC95 % (sección 10). Raza excluida del predictor, lo que no garantiza equidad. |
| Validación | Interna (partición por paciente + CV repetida). **Sin validación temporal ni externa** en UCI; ver notebook MIMIC-IV. |
| Código y registro | `reingresos_utils.py`, este notebook y corridas en MLflow (`mlruns/`). Semilla 42. |


### Riesgo de sesgo (PROBAST)

| Dominio | Riesgo de sesgo | Preocupación de aplicabilidad | Justificación |
|---|---|---|---|
| Participantes | Moderado | Alta | Base retrospectiva con criterios del dataset (diabetes, 1–14 días, laboratorio y medicación). Población de EE. UU. 1999–2008. |
| Predictores | Incierto | Alta | Definición uniforme, pero la disponibilidad al alta no fue verificada; diagnósticos codificados después del alta. |
| Outcome | Alto | Moderada | Reingresos fuera de la red no observados: "NO" puede ser ausencia de registro. Sin etiqueta de evitabilidad. |
| Análisis | Moderado | — | Eventos por parámetro amplios (6,758 eventos en train), CV agrupada repetida, calibración y bootstrap. Test reutilizado de la v1; sin validación externa. |
| Global | Alto | Alta | Determinado por el outcome incompleto y la falta de validación temporal/externa. |
