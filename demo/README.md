# Demo interactiva del modelo v2

Aplicación Streamlit para mostrar en clase cómo el modelo convierte un perfil de alta en una prioridad operativa.

## Inicio rápido

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
streamlit run demo/app.py
```

También puede iniciarse sin activar el entorno:

```bash
./.venv/bin/streamlit run demo/app.py
```

Streamlit abrirá `http://localhost:8501`. La aplicación no requiere Internet ni reentrena el modelo.

## Hugging Face Spaces

La demo está desplegada como un Space Docker privado:

https://huggingface.co/spaces/ldidone/reingresos-diabetes-demo

Para actualizarlo después de cambiar la app o el modelo:

```bash
python3 deploy/huggingface/deploy_space.py
```

La configuración reproducible está en `deploy/huggingface/`.

## Recorrido sugerido para la presentación

1. Empezar con **Riesgo intermedio** y explicar la prevalencia de la cohorte (11,4 %) y el umbral operativo (15,4 %).
2. Cambiar a **Alto riesgo**: el perfil pasa a formar parte del 20 % priorizado.
3. Mover `Internaciones en el último año` de 4 a 0 para mostrar cómo cambia el riesgo en vivo.
4. Abrir **¿Qué está moviendo este resultado?** y aclarar que es un análisis contrafactual local, no causal.
5. Ir a **Impacto operativo**: por cada 1.000 altas, el modelo sigue 200 y capta aproximadamente 45 de 114 reingresos.
6. Cerrar en **Cómo funciona**: LightGBM mejoró poco a la logística y SMOTENC empeoró el desempeño.

## Contenido

- `app.py`: interfaz Streamlit.
- `demo_logic.py`: preparación del perfil, scoring, comparación con LACE/HOSPITAL y análisis “qué pasaría si”.
- `artifacts/readmission_demo.joblib`: modelo LightGBM v2 congelado, recalibrador Platt, modelos de los scores clínicos y distribuciones de referencia.
- `build_demo_model.py`: reconstruye el artefacto desde `data/diabetes.zip` usando exactamente la cohorte y particiones de la v2.

## Reconstruir el modelo

No hace falta para ejecutar la demo. Si cambia el pipeline o se borra el artefacto:

```bash
./.venv/bin/python demo/build_demo_model.py
```

El script comprueba el SHA-256 del dataset, el orden de las 45 variables y que las predicciones no cambien después de serializar. Resultado esperado: AUROC 0,6791 y AP 0,2235 sobre el test de la v2.

## Alcance

Demo académica, no herramienta clínica. No usar datos identificatorios ni tomar decisiones asistenciales. El modelo utiliza datos históricos de EE. UU. (1999–2008), no tiene validación local o temporal y algunas variables pueden no estar disponibles el día del alta.
