# Métricas finales del modelo

## Resumen

El modelo implementado en este proyecto se basa en Isolation Forest para la detección de anomalías en métricas de infraestructura TI.

Durante el desarrollo del prototipo se trabajó con métricas relacionadas con uso de CPU, memoria y disco, procesadas en ventanas temporales para identificar comportamientos fuera del patrón normal.

## Resultados obtenidos

En la fase de evaluación del proyecto se obtuvo una mejora en el desempeño del modelo, alcanzando valores de:

- ROC AUC: 0.8173
- KS: 0.6548

Estos resultados evidencian una mejora en la capacidad del sistema para diferenciar entre comportamientos normales y anómalos.

## Observaciones

Para fines de demostración en el repositorio se utilizó un dataset de ejemplo reducido, por lo que algunos parámetros de ejecución fueron ajustados temporalmente para permitir el entrenamiento y la predicción en un entorno controlado.

En el entorno de trabajo real, el modelo fue concebido para operar con un mayor volumen de datos históricos, lo que permite una detección más robusta y estable.
