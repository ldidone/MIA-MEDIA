---
title: Reingreso hospitalario a 30 días
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Demo académica de priorización de seguimiento posalta
---

# Predicción de reingreso hospitalario a 30 días

Demo interactiva del modelo v2 para priorizar seguimiento después del alta en una cohorte histórica de pacientes con diabetes.

La aplicación permite:

- construir perfiles de alta con casos de ejemplo;
- obtener el riesgo estimado, percentil y prioridad para un cupo del 20 %;
- comparar el modelo con LACE y HOSPITAL aproximados;
- explorar sensibilidades locales no causales;
- traducir el desempeño a impacto operativo por cada 1.000 altas.

## Alcance

Demo académica, no herramienta clínica. El modelo usa datos históricos de 130 hospitales de Estados Unidos entre 1999 y 2008, no tiene validación argentina, temporal ni externa, y no predice evitabilidad o eficacia de una intervención.

Fuente: [Diabetes 130-US Hospitals for Years 1999–2008](https://doi.org/10.24432/C5230J), UCI Machine Learning Repository, CC BY 4.0.
