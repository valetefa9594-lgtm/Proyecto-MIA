# Data - Processed

Esta carpeta contiene los datos procesados y transformados utilizados para el entrenamiento y evaluación del modelo.

## Descripción

Los datos aquí almacenados han pasado por un proceso de:

- Limpieza de valores nulos
- Conversión de tipos de datos
- Agrupación en intervalos de 15 minutos
- Transformación a formato tabular (pivot)
- Imputación de valores faltantes
- Generación de features (ventanas, estadísticas, tendencias)

## Uso

Estos datasets son utilizados directamente por los scripts de entrenamiento y predicción del modelo.

## Nota

Por motivos de confidencialidad, los datos reales no se incluyen en este repositorio.

## Ejemplo de columnas

- t15
- instance
- cpu_usage
- memory_usage
- disk_io
- network_usage
Las direcciones IP y nombres de host fueron sustituidos por identificadores genéricos en los datasets de muestra
