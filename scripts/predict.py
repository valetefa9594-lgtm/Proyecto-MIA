import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
CSV_PATH = "data/sample/ejemplo_dataset.csv"
SEP = ";"

WINDOW_N = 1
SUSTAIN_N = 1

MODELS_DIR = Path("models")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# =========================
# HELPERS
# =========================
def get_ip(instance: str) -> str:
    return str(instance).split(":")[0].strip()


def classify_metric(metric_name: str) -> str:
    m = str(metric_name).lower()
    if "cpu" in m:
        return "CPU"
    if "disk" in m:
        return "DISK"
    if "mem" in m or "memory" in m:
        return "MEMORY"
    return "OTHER"


def sev_label_from_score(score: float) -> str:
    if score > 0.80:
        return "CRITICAL"
    if score > 0.65:
        return "HIGH"
    if score > 0.55:
        return "MEDIUM"
    return "LOW"


# =========================
# PIPELINE: LOAD -> 15m -> PIVOT
# =========================
def load_and_prep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=SEP)
    df["instance"] = df["instance"].astype(str).str.strip()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "value", "instance", "metric"])
    df["t15"] = df["timestamp_utc"].dt.floor("15min")

    interval_df = (
        df.groupby(["t15", "instance", "metric"], as_index=False)["value"]
          .mean()
    )

    wide = interval_df.pivot_table(
        index=["t15", "instance"],
        columns="metric",
        values="value",
        aggfunc="mean"
    ).reset_index()

    wide = wide.sort_values(["instance", "t15"]).reset_index(drop=True)
    wide["ip"] = wide["instance"].apply(get_ip)
    return wide


def enforce_continuity(wide: pd.DataFrame) -> pd.DataFrame:
    full = []
    for srv in wide["instance"].unique():
        temp = wide[wide["instance"] == srv].copy()
        full_range = pd.date_range(temp["t15"].min(), temp["t15"].max(), freq="15min")
        grid = pd.DataFrame({"t15": full_range, "instance": srv})
        temp_full = grid.merge(temp, on=["t15", "instance"], how="left")
        temp_full["ip"] = get_ip(srv)
        full.append(temp_full)

    out = pd.concat(full, ignore_index=True)
    out = out.sort_values(["instance", "t15"]).reset_index(drop=True)
    return out


def impute(wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["t15", "instance", "ip"]
    feat_cols = [c for c in wide.columns if c not in id_cols]

    for c in feat_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    disk_cols = [c for c in feat_cols if "disk" in c.lower()]
    if disk_cols:
        wide[disk_cols] = wide[disk_cols].fillna(0)

    for c in feat_cols:
        if ("cpu" in c.lower()) or ("mem" in c.lower()) or ("memory" in c.lower()):
            wide[c] = wide.groupby("instance")[c].ffill()

    for c in feat_cols:
        if ("cpu" in c.lower()) or ("mem" in c.lower()) or ("memory" in c.lower()):
            wide[c] = wide.groupby("instance")[c].transform(lambda x: x.fillna(x.median()))

    wide[feat_cols] = wide[feat_cols].fillna(wide[feat_cols].median(numeric_only=True))
    return wide


def build_windows(g: pd.DataFrame, window_n: int, instance_name: str = None) -> pd.DataFrame:
    g = g.sort_values("t15").copy()
    g = g.reset_index(drop=True)

    if "instance" not in g.columns:
        g["instance"] = instance_name

    if "ip" not in g.columns:
        g["ip"] = g["instance"].apply(get_ip)

    base_cols = [c for c in g.columns if c not in ["t15", "instance", "ip"]]

    roll = g[base_cols].rolling(window_n, min_periods=window_n)

    out = pd.DataFrame({
        "t15": g["t15"].values,
        "instance": g["instance"].values,
        "ip": g["ip"].values
    })

    out = pd.concat([out, roll.mean().add_prefix("mean_")], axis=1)
    out = pd.concat([out, roll.max().add_prefix("max_")], axis=1)

    # std con ddof=0 para evitar NaN cuando WINDOW_N = 1
    std_df = roll.std(ddof=0).add_prefix("std_").fillna(0)
    out = pd.concat([out, std_df], axis=1)

    trend = g[base_cols].diff().rolling(window_n, min_periods=1).mean()
    trend = trend.add_prefix("trend_").fillna(0)
    out = pd.concat([out, trend], axis=1)

    return out.reset_index(drop=True)

def dominant_feature(row, numeric_feature_cols, normal_means):
    vals = pd.to_numeric(row[numeric_feature_cols], errors="coerce")
    diffs = (vals - normal_means).abs().dropna()
    if diffs.empty:
        return None
    return diffs.astype(float).idxmax()


# =========================
# MAIN
# =========================
def main():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    iso = joblib.load(MODELS_DIR / "isoforest.joblib")

    with open(MODELS_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(MODELS_DIR / "threshold.json", "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    wide = load_and_prep(CSV_PATH)
    wide = enforce_continuity(wide)
    wide = impute(wide)

    win = (
        wide.groupby("instance", group_keys=False)
            .apply(lambda x: build_windows(x, WINDOW_N, instance_name=x.name))
            .reset_index(drop=True)
    )

    if win.empty:
        raise ValueError("No se generaron ventanas para predicción.")

    id_cols = ["t15", "instance", "ip"]

    missing_features = [c for c in feature_names if c not in win.columns]
    for c in missing_features:
        win[c] = 0.0

    X = win[feature_names].copy().fillna(0)
    X_s = scaler.transform(X)
    scores = -iso.score_samples(X_s)

    normal_means = X.mean(numeric_only=True)

    final = win[id_cols].copy()
    final["server"] = final["instance"]
    final["status"] = "active"
    final["anomaly_score"] = scores
    final["threshold"] = threshold
    final["is_anomaly"] = final["anomaly_score"] >= final["threshold"]
    final["is_sustained"] = (
        final.groupby("instance")["is_anomaly"]
             .transform(lambda s: s.rolling(SUSTAIN_N, min_periods=1).sum() >= SUSTAIN_N)
    )

    final["dominant_metric"] = win.apply(
        lambda r: dominant_feature(r, feature_names, normal_means),
        axis=1
    )

    final["resource"] = final["dominant_metric"].apply(classify_metric)
    final["severity"] = final["anomaly_score"].apply(sev_label_from_score)

    events_out = final[final["is_sustained"] == True].copy()

    final.to_csv(OUT_DIR / "final_scores.csv", index=False)
    events_out.to_csv(OUT_DIR / "events.csv", index=False)

    last_state = (
        final.sort_values("t15")
             .groupby("server", as_index=False)
             .tail(1)[["t15", "server", "status", "anomaly_score", "is_anomaly", "is_sustained", "threshold"]]
    )
    last_state.to_csv(OUT_DIR / "alerts_last.csv", index=False)

    print("✅ Predicción completada")
    print("Registros procesados:", len(final))
    print("Eventos detectados:", len(events_out))


if __name__ == "__main__":
    main()