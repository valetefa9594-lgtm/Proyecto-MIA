def get_ip(instance: str) -> str:
    """
    Extrae la IP desde un valor tipo '192.168.1.10:9182'.
    """
    return str(instance).split(":")[0].strip()


def classify_metric(metric_name: str) -> str:
    """
    Clasifica una métrica en una categoría general.
    """
    m = str(metric_name).lower()

    if "cpu" in m:
        return "CPU"
    if "disk" in m:
        return "DISK"
    if "mem" in m or "memory" in m:
        return "MEMORY"

    return "OTHER"


def simplify_resource(metric_name: str) -> str:
    """
    Simplifica la categoría de recurso para visualización o alertas.
    """
    return classify_metric(metric_name)


def sev_label_from_score(score: float) -> str:
    """
    Convierte un anomaly score en etiqueta de severidad.
    """
    if score > 0.80:
        return "CRITICAL"
    if score > 0.65:
        return "HIGH"
    if score > 0.55:
        return "MEDIUM"
    return "LOW"


def sev_value(level: str) -> int:
    """
    Convierte la severidad a valor numérico.
    """
    mapping = {
        "NONE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4
    }
    return mapping.get(level, 0)
