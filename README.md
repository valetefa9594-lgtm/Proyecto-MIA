# Sistema Predictivo de Detección de Fallas en Infraestructura TI mediante IA

## 1. Descripción del problema

La organización analizada depende del funcionamiento continuo de su infraestructura tecnológica para soportar procesos críticos del negocio. Sin embargo, el monitoreo tradicional basado en umbrales fijos y revisión reactiva de alertas no permite detectar de forma temprana comportamientos anómalos en los servidores. Como consecuencia, las fallas suelen identificarse cuando el sistema ya presenta degradación crítica, generando lentitud, interrupciones del servicio y afectaciones operativas.

## 2. Solución propuesta

Este prototipo implementa un sistema de detección temprana de anomalías en servidores de infraestructura TI mediante inteligencia artificial. La solución utiliza un modelo de aprendizaje automático no supervisado, específicamente Isolation Forest, para analizar métricas históricas de CPU, memoria y disco, identificar patrones anómalos y generar alertas sostenidas.

El sistema contempla:
- carga y limpieza de métricas,
- continuidad temporal por intervalos de 15 minutos,
- imputación de datos faltantes,
- construcción de variables de ventana,
- entrenamiento del modelo,
- cálculo de umbral de anomalía,
- predicción sobre nuevos datos,
- detección de eventos sostenidos,
- exportación de resultados y alertas.

## 3. Estructura del repositorio

- `data/`: datasets de entrenamiento, predicción y muestras.
- `scripts/`: scripts principales del sistema.
- `models/`: artefactos del modelo entrenado.
- `results/`: resultados finales, métricas y evidencias.
- `notebooks/`: análisis exploratorio, entrenamiento y validación.
- `docs/`: anexos, diagramas y documentación de apoyo.
- `state/`: archivos de persistencia de eventos.

## 4. Requisitos técnicos y dependencias

- Python 3.10 o superior
- pandas
- numpy
- scikit-learn
- joblib

Instalación de dependencias:

```bash
pip install -r requirements.txt
