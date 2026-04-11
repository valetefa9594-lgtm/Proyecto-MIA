import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# =========================
# CONFIG (15 MIN REAL)
# =========================
CSV_PATH = "data/sample/ejemplo_dataset.csv"   
SEP = ";"

WINDOW_N = 32          #8 horas
MIN_POINTS = 192       # 192 x 15min = 48 horas warm-up
SUSTAIN_N = 8          # 8 x 15min = 2 horas sostenida

PERCENTILE = 97        # threshold = P97 sobre normales
CONTAM = 0.01          # isolation forest

# UNICO servidor normal
NORMAL_INSTANCES = ["192.168.60.12:9182","192.168.60.45:9182","10.20.0.5:9182","192.168.60.53:9182"]

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Prometheus textfile collector (node_exporter)
PROM_FILE = "/var/lib/node_exporter/textfile_collector/ia_alerts.prom"
TMP_FILE = PROM_FILE + ".tmp"

# Persistencia del contador (para ia_event_start_total)
STATE_DIR = Path("state")  # puedes cambiar a /root/ia/modelado/state si quieres
STATE_DIR.mkdir(exist_ok=True, parents=True)
EVENT_COUNTER_FILE = STATE_DIR / "event_start_counts.json"


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


def simplify_resource(metric_name: str) -> str:
    # Para tesis: CPU / MEMORY / DISK
    return classify_metric(metric_name)


def sev_label_from_score(score: float) -> str:
    # Igual que tu lógica
    if score > 0.80:
        return "CRITICAL"
    if score > 0.65:
        return "HIGH"
    if score > 0.55:
        return "MEDIUM"
    return "LOW"


def sev_value(level: str) -> int:
    # LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4, NONE=0
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(level, 0)


def load_event_counter() -> dict:
    """Carga el contador persistente por servidor (exported_instance)."""
    if EVENT_COUNTER_FILE.exists():
        try:
            with open(EVENT_COUNTER_FILE, "r") as f:
                data = json.load(f)
            return {k: int(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def save_event_counter(counter: dict) -> None:
    """Guarda el contador de forma atómica."""
    tmp = str(EVENT_COUNTER_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: int(v) for k, v in counter.items()}, f, indent=2, sort_keys=True)
    os.replace(tmp, EVENT_COUNTER_FILE)


def increment_event_counter(counter: dict, event_start_latest: dict) -> dict:
    """
    Suma +1 al contador cuando event_start_latest[server]==1
    event_start_latest: {exported_instance: 0/1}
    """
    for srv, v in event_start_latest.items():
        if int(v) == 1:
            counter[srv] = int(counter.get(srv, 0)) + 1
        else:
            counter.setdefault(srv, int(counter.get(srv, 0)))
    return counter


# =========================
# PIPELINE: LOAD -> 15m -> PIVOT
# =========================
def load_and_prep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=SEP)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp_utc", "value", "instance", "metric"])

    # bucket 15m
    df["t15"] = df["timestamp_utc"].dt.floor("15min")

    # agregación
    interval_df = (
        df.groupby(["t15", "instance", "metric"], as_index=False)["value"]
          .mean()
    )

    # pivot ancho
    wide = interval_df.pivot_table(
        index=["t15", "instance"],
        columns="metric",
        values="value",
        aggfunc="mean"
    ).reset_index()

    wide = wide.sort_values(["instance", "t15"]).reset_index(drop=True)
    wide["ip"] = wide["instance"].apply(get_ip)
    return wide


# =========================
# CONTINUIDAD (15m) POR SERVIDOR
# =========================
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


# =========================
# IMPUTACION ROBUSTA
# =========================
def impute(wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["t15", "instance", "ip"]
    feat_cols = [c for c in wide.columns if c not in id_cols]

    # asegurar numérico
    for c in feat_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    # discos inexistentes -> 0
    disk_cols = [c for c in feat_cols if "disk" in c.lower()]
    if disk_cols:
        wide[disk_cols] = wide[disk_cols].fillna(0)

    # cpu/mem -> ffill por servidor
    for c in feat_cols:
        if ("cpu" in c.lower()) or ("mem" in c.lower()) or ("memory" in c.lower()):
            wide[c] = wide.groupby("instance")[c].ffill()

    # cpu/mem al inicio -> mediana por servidor
    for c in feat_cols:
        if ("cpu" in c.lower()) or ("mem" in c.lower()) or ("memory" in c.lower()):
            wide[c] = wide.groupby("instance")[c].transform(lambda x: x.fillna(x.median()))

    # resto -> mediana global
    wide[feat_cols] = wide[feat_cols].fillna(wide[feat_cols].median(numeric_only=True))
    return wide


# =========================
# WINDOWS FEATURES (rolling window)
# =========================
def build_windows(g: pd.DataFrame, window_n: int) -> pd.DataFrame:
    g = g.sort_values("t15").copy()
    base_cols = [c for c in g.columns if c not in ["t15", "instance", "ip"]]

    roll = g[base_cols].rolling(window_n, min_periods=window_n)

    out = pd.DataFrame({"t15": g["t15"], "instance": g["instance"], "ip": g["ip"]})
    out = pd.concat([out, roll.mean().add_prefix("mean_")], axis=1)
    out = pd.concat([out, roll.max().add_prefix("max_")], axis=1)
    out = pd.concat([out, roll.std().add_prefix("std_")], axis=1)

    trend = g[base_cols].diff().rolling(window_n, min_periods=window_n).mean()
    out = pd.concat([out, trend.add_prefix("trend_")], axis=1)

    return out.dropna().reset_index(drop=True)


# =========================
# DOMINANT METRIC (numerica)
# =========================
def dominant_feature(row, numeric_feature_cols, normal_means):
    vals = pd.to_numeric(row[numeric_feature_cols], errors="coerce")
    diffs = (vals - normal_means).abs()
    diffs = diffs.dropna()
    if diffs.empty:
        return None
    diffs = diffs.astype(float)
    return diffs.idxmax()


# =========================
# MAIN
# =========================
def main():
    # 1) cargar y preparar
    wide = load_and_prep(CSV_PATH)
    wide = enforce_continuity(wide)
    wide = impute(wide)

    # 2) ventanas por servidor
    win = (
        wide.groupby("instance", group_keys=False)
            .apply(lambda x: build_windows(x, WINDOW_N))
            .reset_index(drop=True)
    )

    # =========================
    # 3) ENTRENAR CON NORMAL
    # =========================
    train_df = win[win["instance"].isin(NORMAL_INSTANCES)].copy()
    if train_df.empty:
        raise ValueError(f"No existe el servidor normal {NORMAL_INSTANCES} en el dataset.")

    n_points = train_df["t15"].nunique()
    if n_points < MIN_POINTS:
        raise ValueError(
            f"Servidor normal tiene solo {n_points} puntos. "
            f"Necesitas MIN_POINTS={MIN_POINTS} para entrenar."
        )

    id_cols = ["t15", "instance", "ip"]

    numeric_cols = win.drop(columns=id_cols).select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No hay columnas numéricas para entrenar. Revisa tu CSV y el pivot.")

    X_train = train_df[numeric_cols].copy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=500,
        contamination=CONTAM,
        random_state=42
    )
    iso.fit(X_train_s)

    train_scores = -iso.score_samples(X_train_s)
    threshold = float(np.percentile(train_scores, PERCENTILE))

    print(f"✅ Entrenado con NORMAL={NORMAL_INSTANCES[0]}")
    print(f"✅ Threshold global P{PERCENTILE}: {threshold:.6f}")

    normal_means = train_df[numeric_cols].mean(numeric_only=True)

    # =========================
    # 4) INFERENCIA EN TODOS
    # =========================
    results = []
    for server in win["instance"].unique():
        data = win[win["instance"] == server].sort_values("t15").reset_index(drop=True)

        if data["t15"].nunique() < MIN_POINTS:
            tmp = data[id_cols].copy()
            tmp["server"] = server
            tmp["status"] = "learning"
            tmp["anomaly_score"] = np.nan
            tmp["threshold"] = threshold
            tmp["is_anomaly"] = False
            tmp["is_sustained"] = False
            results.append(tmp)
            continue

        X = data[numeric_cols].copy()
        X_s = scaler.transform(X)
        scores = -iso.score_samples(X_s)

        out = data[id_cols].copy()
        out["server"] = server
        out["status"] = "active"
        out["anomaly_score"] = scores
        out["threshold"] = threshold

        out["is_anomaly"] = out["anomaly_score"] >= threshold
        out["is_sustained"] = out["is_anomaly"].rolling(SUSTAIN_N).sum() >= SUSTAIN_N

        results.append(out)

    final = pd.concat(results, ignore_index=True)

    # =========================
    # 5) EVENTOS (solo sostenidas) + dominant_metric
    # =========================
    alerts = final[final["is_sustained"] == True].copy()

    if not alerts.empty:
        alerts_full = alerts.merge(
            win,
            left_on=["t15", "server"],
            right_on=["t15", "instance"],
            how="left",
            suffixes=("", "_feat")
        )

        alerts_full["dominant_metric"] = alerts_full.apply(
            lambda r: dominant_feature(r, numeric_cols, normal_means),
            axis=1
        )

        alerts_full["resource"] = alerts_full["dominant_metric"].apply(classify_metric)
        alerts_full["severity"] = alerts_full["anomaly_score"].apply(sev_label_from_score)

        alerts_full = alerts_full.sort_values(["server", "t15"]).copy()
        alerts_full["event_start"] = (
            alerts_full.groupby("server")["is_sustained"]
                .apply(lambda s: s & (~s.shift(1, fill_value=False)))
                .reset_index(level=0, drop=True)
        )

        events = alerts_full[alerts_full["event_start"] == True].copy()
        events_out = events[["t15", "server", "resource", "dominant_metric", "severity", "anomaly_score", "threshold"]].copy()
    else:
        alerts_full = pd.DataFrame()
        events_out = pd.DataFrame(columns=["t15", "server", "resource", "dominant_metric", "severity", "anomaly_score", "threshold"])

    # =========================
    # 6) OUTPUTS CSV
    # =========================
    final.to_csv(OUT_DIR / "final_scores.csv", index=False)
    events_out.to_csv(OUT_DIR / "events.csv", index=False)

    last_state = (
        final.sort_values("t15")
             .groupby("server", as_index=False)
             .tail(1)[["t15", "server", "status", "anomaly_score", "is_anomaly", "is_sustained", "threshold"]]
    )
    last_state.to_csv(OUT_DIR / "alerts_last.csv", index=False)

    # =========================
    # 7) EVENT START LATEST (para Prometheus)
    # =========================
    event_start_latest = {}
    for srv, g in final.sort_values("t15").groupby("server"):
        gg = g.tail(2).copy()
        if len(gg) < 2:
            event_start_latest[srv] = 0
            continue
        prev_s = bool(gg.iloc[0]["is_sustained"])
        curr_s = bool(gg.iloc[1]["is_sustained"])
        event_start_latest[srv] = 1 if (curr_s and (not prev_s)) else 0

    # =========================
    # 7.1) EVENT START TOTAL (contador persistente)
    # =========================
    event_counter = load_event_counter()
    event_counter = increment_event_counter(event_counter, event_start_latest)
    save_event_counter(event_counter)

    # =========================
    # 8) EXPORT TO PROMETHEUS TEXTFILE
    # =========================
    with open(TMP_FILE, "w") as f:
        f.write("# HELP ia_anomaly_score IsolationForest anomaly score\n")
        f.write("# TYPE ia_anomaly_score gauge\n")
        f.write("# HELP ia_anomaly_sustained 1 if anomaly sustained else 0\n")
        f.write("# TYPE ia_anomaly_sustained gauge\n")
        f.write("# HELP ia_dominant_resource Dominant resource category when anomaly sustained\n")
        f.write("# TYPE ia_dominant_resource gauge\n")
        f.write("# HELP ia_anomaly_severity Severity level label for anomalies\n")
        f.write("# TYPE ia_anomaly_severity gauge\n")
        f.write("# HELP ia_anomaly_severity_value Severity as numeric (LOW=1..CRITICAL=4)\n")
        f.write("# TYPE ia_anomaly_severity_value gauge\n")
        f.write("# HELP ia_event_start 1 when sustained anomaly event starts (latest transition)\n")
        f.write("# TYPE ia_event_start gauge\n")
        f.write("# HELP ia_event_start_total Total sustained anomaly event starts (counter)\n")
        f.write("# TYPE ia_event_start_total counter\n\n")

        # 1) métricas por servidor (último estado)
        for _, row in last_state.iterrows():
            exported_instance = row["server"]
            score = 0.0 if pd.isna(row["anomaly_score"]) else float(row["anomaly_score"])
            sustained = 1 if bool(row["is_sustained"]) else 0

            dominant_res = "NONE"
            level = "NONE"
            level_num = 0

            if sustained == 1 and not events_out.empty:
                ev = (
                    events_out[events_out["server"] == exported_instance]
                    .sort_values("t15")
                    .tail(1)
                )
                if not ev.empty:
                    dom_metric = ev.iloc[0]["dominant_metric"]
                    dominant_res = simplify_resource(dom_metric)
                    level = str(ev.iloc[0]["severity"])
                    level_num = sev_value(level)
                else:
                    level = sev_label_from_score(score)
                    level_num = sev_value(level)
            elif sustained == 1:
                level = sev_label_from_score(score)
                level_num = sev_value(level)

            ev_start = int(event_start_latest.get(exported_instance, 0))
            ev_total = int(event_counter.get(exported_instance, 0))

            f.write(f'ia_anomaly_score{{exported_instance="{exported_instance}"}} {score}\n')
            f.write(f'ia_anomaly_sustained{{exported_instance="{exported_instance}"}} {sustained}\n')
            f.write(f'ia_dominant_resource{{exported_instance="{exported_instance}",resource="{dominant_res}"}} 1\n')
            f.write(f'ia_anomaly_severity{{exported_instance="{exported_instance}",level="{level}"}} 1\n')
            f.write(f'ia_anomaly_severity_value{{exported_instance="{exported_instance}"}} {level_num}\n')
            f.write(f'ia_event_start{{exported_instance="{exported_instance}"}} {ev_start}\n')
            f.write(f'ia_event_start_total{{exported_instance="{exported_instance}"}} {ev_total}\n')

    os.replace(TMP_FILE, PROM_FILE)

    print("✅ OK")
    print("Eventos:", len(events_out))
    print("Estado por servidor:", len(last_state))


if __name__ == "__main__":
    main()
