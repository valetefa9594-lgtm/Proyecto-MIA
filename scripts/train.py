# train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
CSV_PATH = "data/sample/ejemplo_dataset.csv"
SEP = ";"

# 15m real
WINDOW_N = 32      # 8 horas
MIN_POINTS = 1
P = 97
CONTAM = 0.01

NORMAL_IPS = ["192.168.1.2", "192.168.1.5"] 

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_ip(instance: str) -> str:
    return str(instance).split(":")[0].strip()

def load_and_prep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=SEP)
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
    return wide

def enforce_continuity_15m(wide: pd.DataFrame) -> pd.DataFrame:
    full_data = []
    for srv in wide["instance"].unique():
        temp = wide[wide["instance"] == srv].copy()
        full_range = pd.date_range(temp["t15"].min(), temp["t15"].max(), freq="15min")
        grid = pd.DataFrame({"t15": full_range, "instance": srv})
        temp_full = grid.merge(temp, on=["t15", "instance"], how="left")
        full_data.append(temp_full)
    wide2 = pd.concat(full_data, ignore_index=True)
    wide2 = wide2.sort_values(["instance", "t15"]).reset_index(drop=True)
    return wide2

def impute_post_continuity(wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["t15", "instance"]
    feature_cols = [c for c in wide.columns if c not in id_cols]

    disk_cols = [c for c in feature_cols if "disk" in c.lower()]
    if disk_cols:
        wide[disk_cols] = wide[disk_cols].fillna(0)

    for col in feature_cols:
        if "cpu" in col.lower() or "mem" in col.lower():
            wide[col] = wide.groupby("instance")[col].ffill()

    for col in feature_cols:
        if "cpu" in col.lower() or "mem" in col.lower():
            wide[col] = wide.groupby("instance")[col].transform(lambda x: x.fillna(x.median()))

    wide[feature_cols] = wide[feature_cols].fillna(wide[feature_cols].median(numeric_only=True))

    return wide


def build_windows(g: pd.DataFrame, window_n: int, instance_name: str = None) -> pd.DataFrame:
    g = g.sort_values("t15").copy()
    g = g.reset_index(drop=True)

    if "instance" not in g.columns:
        g["instance"] = instance_name

    feat_cols = [c for c in g.columns if c not in ["t15", "instance"]]

    roll = g[feat_cols].rolling(window_n, min_periods=window_n)

    out = pd.DataFrame({
        "t15": g["t15"].values,
        "instance": g["instance"].values
    })

    out = pd.concat([out, roll.mean().add_prefix("mean_")], axis=1)
    out = pd.concat([out, roll.max().add_prefix("max_")], axis=1)

    std_df = roll.std(ddof=0).add_prefix("std_").fillna(0)
    out = pd.concat([out, std_df], axis=1)

    trend = g[feat_cols].diff().rolling(window_n, min_periods=1).mean()
    trend = trend.add_prefix("trend_").fillna(0)
    out = pd.concat([out, trend], axis=1)

    return out.reset_index(drop=True)

def main():
    wide = load_and_prep(CSV_PATH)
    wide = enforce_continuity_15m(wide)
    wide = impute_post_continuity(wide)

    wide = wide.reset_index(drop=True)

    print("Columnas de wide:", wide.columns)
    print(wide.head())


    
    win = (
    wide.groupby("instance", group_keys=False)
        .apply(lambda x: build_windows(x, WINDOW_N, instance_name=x.name))
        .reset_index(drop=True)
)

    win = win.copy()
    win["ip"] = win["instance"].apply(get_ip)

    train_df = win[win["ip"].isin(NORMAL_IPS)].copy()

    points_count = train_df.groupby("instance")["t15"].nunique()
    good_instances = points_count[points_count >= MIN_POINTS].index.tolist()
    train_df = train_df[train_df["instance"].isin(good_instances)].copy()

    if train_df.empty:
        raise ValueError("No hay suficientes datos para entrenar con NORMAL_IPS y MIN_POINTS.")

    id_cols = ["t15", "instance", "ip"]
    X_train = train_df.drop(columns=id_cols)

    feature_names = X_train.columns.tolist()
    (MODELS_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2)
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=500,
        contamination=CONTAM,
        random_state=42
    )
    iso.fit(X_train_s)

    train_scores = -iso.score_samples(X_train_s)
    threshold = float(np.percentile(train_scores, P))

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(iso, MODELS_DIR / "isoforest.joblib")

    (MODELS_DIR / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "P": P, "CONTAM": CONTAM})
    )
    (MODELS_DIR / "good_instances.json").write_text(
        json.dumps({"good_instances": good_instances})
    )
    (MODELS_DIR / "config.json").write_text(json.dumps({
        "WINDOW_N": WINDOW_N,
        "MIN_POINTS": MIN_POINTS,
        "P": P,
        "CONTAM": CONTAM,
        "NORMAL_IPS": NORMAL_IPS
    }))

    print("✅ Entrenado OK")
    print("   good_instances:", good_instances)
    print(f"   threshold (P{P}): {threshold:.6f}")

if __name__ == "__main__":
    main()
