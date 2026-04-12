# Modelo de IA para la detección temprana de fallas en servidores

## Descripción

En entornos de infraestructura tecnológica, el monitoreo tradicional basado en umbrales fijos presenta limitaciones para detectar anomalías de forma oportuna. Este proyecto propone un modelo de inteligencia artificial basado en Isolation Forest para identificar comportamientos anómalos en métricas de servidores como CPU, memoria y disco.

El objetivo es anticipar fallas antes de que se conviertan en incidentes críticos, mejorando la disponibilidad de los servicios tecnológicos.

## Solución

Se implementa un sistema de detección de anomalías basado en aprendizaje automático no supervisado que:

- Procesa métricas de infraestructura
- Genera variables mediante ventanas temporales
- Detecta anomalías usando Isolation Forest
- Genera alertas basadas en anomalías sostenidas (2 horas)

## Arquitectura del sistema

![Arquitectura](docs/images/figura_7_arquitectura.p)

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
