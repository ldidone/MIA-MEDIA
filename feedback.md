# **Devolución: Predicción de reingreso hospitalario antes de 30 días / Lucas Didone**

*Fundamentos de Medicina y Aplicaciones de IA (CEIA, FIUBA). Devolución de la primera entrega, octubre de 2026\.*  
Es el trabajo metodológicamente más sólido de este grupo. La cohorte está bien definida, la partición es por paciente, el test quedó reservado y las conclusiones se limitan a lo que los datos permiten afirmar. Las observaciones de abajo apuntan al paso de prueba de concepto a herramienta implementable.

## **1\. Análisis de Factibilidad**

La factibilidad técnica está demostrada: el pipeline corre en CPU, es reproducible (semilla, versiones, SHA-256 del dataset) y detecta señal fuera de los pacientes de entrenamiento. Hay decisiones que muestran mucho criterio, por ejemplo:  
*"Selección previa: logística por AP de validación (0,219 frente a 0,216 de boosting). En test, boosting tiene AP 0,220, pero se conserva la decisión tomada en validación."*  
Respetar la decisión de validación aunque el test favorezca a otro modelo es exactamente lo correcto. También lo es excluir fallecimientos y hospice (destinos 11, 13, 14, 19, 20 y 21), dejar *race* y *payer\_code* fuera del predictor y correr el control con etiquetas permutadas.  
Los cuellos de botella están en el uso real:

> * **Transportabilidad.** Datos de EE. UU. de 1999 a 2008, con CIE-9 y otro sistema de salud. Un AUROC de 0,67 no se puede trasladar a un hospital argentino actual.  
> * **Disponibilidad de las variables al alta.** El trabajo lo señala con precisión: *"Se predice al alta, cuando la estancia, la medicación y los diagnósticos de la internación pueden estar disponibles. Esa disponibilidad requiere verificación en una futura implementación."* En muchas instituciones la codificación de diagnósticos se completa días después del egreso. Si el dato no existe al momento de decidir, el modelo no se puede usar en ese momento. Es la primera pregunta a responder con un hospital.  
> * **Costo de las falsas alertas.** Con *"Precision 22,15%, recall 38,78% y lift 1,94"*, de cada 9 altas seleccionadas unas 7 no reingresan. Es razonable si la intervención es barata (un llamado telefónico), no si es una visita domiciliaria. La elección del cupo de 20 % debería justificarse con el costo de la intervención.  
> * **Estabilidad de las estimaciones.** En validación el modelo elegido tiene AUROC 0,653 y en test 0,672. Que el test dé mejor que la validación no es un problema en sí, pero muestra cuánto varía la métrica entre particiones. Eso refuerza el paso que ya planean: validación agrupada repetida para estimar estabilidad.

## **2\. Vinculación con el Material de Clase**

**Bien abordado:**

> * **Investigación clínica con IA (Nava).** El trabajo toma en serio la crítica a la real-world evidence de la clase: reconoce que la etiqueta "NO" puede significar *"ausencia de reingreso observado, atención fuera de la red o seguimiento incompleto"*, que no hay etiqueta de evitabilidad y que el experimento no prueba que el seguimiento reduzca reingresos. La pregunta operativa de la presentación ("¿a qué altas priorizar si el equipo puede seguir solo al 20 %?") conecta con los bloques de estratificación de pacientes y proyección de recursos de esa clase.  
> * **Métricas para clases desbalanceadas.** AP contra prevalencia, Brier, recall al 20 % y lift, con criterios de continuidad fijados antes de abrir el test.

**Faltó o requiere revisión:**

> * **Ecosistema Digital en Salud (Rosa).** Para llevar el modelo a un hospital local hay que mapear los diagnósticos a CIE-10 o SNOMED CT y definir de qué sistema (HCE, sistema de egresos, facturación) saldría cada una de las 42 variables. La interoperabilidad semántica de la clase es el puente entre el dataset y la implementación.  
> * **Comparación con lo que ya se usa.** El baseline "solo utilización previa" es un buen control, pero falta comparar con los scores clínicos de reingreso existentes (índice LACE, HOSPITAL score). Sin esa referencia no se sabe si el modelo aporta algo frente a una herramienta simple que un equipo podría calcular a mano.  
> * **Impacto.** La clase de investigación clínica sugiere el paso siguiente: un ensayo pragmático que aleatorice el seguimiento guiado por el modelo frente al seguimiento habitual.  
> * **HbA1c.** El paper de referencia del dataset (Strack et al., 2014\) trata justamente sobre la medición de HbA1c y el reingreso. Vale la pena comentar qué rol juega A1Cresult en el modelo, ya que la conservan como categoría "prueba no realizada".

## **3\. Recomendaciones Técnicas**

> * **Datos y Bases de Datos:** MIMIC-IV (PhysioNet, acceso con credencial y curso CITI) es el paso natural: admisiones entre 2008 y 2019, diagnósticos con CIE-9 y CIE-10, y resúmenes de alta en MIMIC-IV-Note. Su variable *anchor\_year\_group* permite la validación temporal que la versión pública de UCI no permite, como ustedes mismos señalan. eICU-CRD sirve para analizar heterogeneidad entre hospitales. Para la versión local, armar con el área de estadística de un hospital argentino un diccionario de datos mínimo que indique en qué momento está disponible cada campo.  
> * **Modelos y Algoritmos:** Agregar LACE y HOSPITAL score como baselines (aproximados con las variables de UCI, completos en MIMIC). Mantener logística frente a gradient boosting y sumar análisis de curva de decisión, recalibración en validación (isotónica o Platt) y validación cruzada agrupada por paciente y repetida. Como extensión con MIMIC-IV-Note, representar el resumen de alta con un modelo clínico de lenguaje (Bio\_ClinicalBERT u otro más reciente de contexto largo) y medir la ganancia sobre las variables tabulares. Reportar métricas por subgrupo con intervalos, como ya anticipan.  
> * **Herramientas e Infraestructura:** El stack actual es ejemplar. Agregar dcurves para las curvas de decisión, un registro de corridas (MLflow o similar) y redactar la ficha del modelo siguiendo TRIPOD+AI, con evaluación de riesgo de sesgo según PROBAST.

## **4\. Próximos Pasos (Action Items)**

> 1. Agregar LACE y HOSPITAL score como baselines y una curva de decisión con el costo de seguimiento explícito, para justificar el cupo del 20 %.  
> 2. Replicar el pipeline en MIMIC-IV con partición temporal por *anchor\_year\_group* y medir cuánto se degrada el desempeño en el tiempo.  
> 3. Con un hospital local, relevar qué variables están disponibles el día del alta y en qué sistema, y mapear los diagnósticos a CIE-10 o SNOMED CT.  
> 4. Esbozar para la entrega final el protocolo de un ensayo pragmático (intervención de seguimiento sobre el 20 % priorizado, outcome primario: reingreso a 30 días). Como detalle formal, corregir la errata de la diapositiva 4 de la presentación ("internación.z").