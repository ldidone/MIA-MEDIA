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

## Fuentes y licencia del dataset

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999–2008. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. Licencia CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.

Dataset redistribuido sin modificar dentro de `data/diabetes.zip`. Los filtros y transformaciones están documentados en el notebook. No se implica aval de los autores.

Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. https://doi.org/10.1155/2014/781670.

Todos los resultados del informe y las slides proceden de la ejecución incluida, no de métricas publicadas por terceros.
