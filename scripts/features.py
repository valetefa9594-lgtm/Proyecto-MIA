import pandas as pd
import numpy as np


def build_windows(g: pd.DataFrame, window_n: int) -> pd.DataFrame:
    """
    Genera features rolling por servidor:
    - media
    - máximo
    - desviación estándar
    - tendencia promedio
    """
    g = g.sort_values("t15").copy()

    base_cols = [c for c in g.columns if c not in ["t15", "instance", "ip"]]

    roll = g[base_cols].rolling(window_n, min_periods=window_n)

    out = pd.DataFrame({
        "t15": g["t15"],
        "instance": g["instance"]
    })

    if "ip" in g.columns:
        out["ip"] = g["ip"]

    out = pd.concat([out, roll.mean().add_prefix("mean_")], axis=1)
    out = pd.concat([out, roll.max().add_prefix("max_")], axis=1)
    out = pd.concat([out, roll.std().add_prefix("std_")], axis=1)

    trend = g[base_cols].diff().rolling(window_n, min_periods=window_n).mean()
    out = pd.concat([out, trend.add_prefix("trend_")], axis=1)

    return out.dropna().reset_index(drop=True)


def build_features_by_instance(wide: pd.DataFrame, window_n: int) -> pd.DataFrame:
    """
    Aplica la generación de ventanas por cada servidor.
    """
    win = (
        wide.groupby("instance", group_keys=False)
            .apply(lambda x: build_windows(x, window_n))
            .reset_index(drop=True)
    )
    return win


def dominant_feature(row: pd.Series, numeric_feature_cols: list, normal_means: pd.Series):
    """
    Devuelve la feature numérica con mayor desviación absoluta respecto
    al promedio normal.
    """
    vals = pd.to_numeric(row[numeric_feature_cols], errors="coerce")
    diffs = (vals - normal_means).abs()
    diffs = diffs.dropna()

    if diffs.empty:
        return None

    diffs = diffs.astype(float)
    return diffs.idxmax()
