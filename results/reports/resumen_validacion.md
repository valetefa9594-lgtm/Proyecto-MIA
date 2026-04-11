# Resumen de validación del prototipo

## Objetivo

Validar el funcionamiento del pipeline de detección de anomalías en infraestructura TI mediante un flujo compuesto por entrenamiento, predicción y generación de resultados.

## Componentes validados

Se verificó la ejecución de los siguientes scripts:

- `scripts/train.py`
- `scripts/predict.py`

## Resultados generados

Durante la ejecución del sistema se generaron los siguientes archivos de salida:

- `models/isoforest.joblib`
- `models/scaler.joblib`
- `models/feature_names.json`
- `models/threshold.json`
- `out/final_scores.csv`
- `out/events.csv`
- `out/alerts_last.csv`

## Observaciones de validación

El prototipo fue probado con un dataset de ejemplo de tamaño reducido, por lo que ciertos parámetros como el tamaño de ventana y el mínimo de puntos requeridos fueron ajustados únicamente para fines de demostración.

Esto permitió comprobar la lógica general del sistema, así como la correcta integración entre el preprocesamiento, la generación de variables, el entrenamiento del modelo y la detección de anomalías.

## Conclusión

La validación realizada permitió confirmar que el prototipo ejecuta correctamente el flujo completo de entrenamiento y predicción, generando resultados utilizables para análisis posterior y visualización.
