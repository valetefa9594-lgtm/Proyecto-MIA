# Manual técnico del prototipo

## 1. Descripción general

Este prototipo implementa un sistema de detección de anomalías en métricas de infraestructura TI mediante técnicas de aprendizaje automático.

El flujo general del sistema contempla:

- carga de métricas desde archivo CSV
- preprocesamiento y transformación temporal
- generación de variables mediante ventanas deslizantes
- entrenamiento de modelo Isolation Forest
- predicción de anomalías
- generación de archivos de salida para análisis posterior

## 2. Estructura del repositorio

- `data/`: datasets de ejemplo y documentación de datos
- `notebooks/`: análisis exploratorio, preprocesamiento, entrenamiento y evaluación
- `scripts/`: scripts ejecutables del sistema
- `models/`: artefactos generados durante el entrenamiento
- `results/`: resultados, métricas y reportes
- `docs/`: documentación técnica y de apoyo

## 3. Requisitos

Para ejecutar el proyecto se requiere:

- Python 3.11 o 3.12 recomendado
- Instalación de dependencias desde `requirements.txt`

Comando de instalación:

```bash
pip install -r requirements.txt
