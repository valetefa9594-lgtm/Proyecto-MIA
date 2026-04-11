import os
import json
from pathlib import Path

import pandas as pd

# =========================
# CONFIG
# =========================
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

PROM_FILE = "/var/lib/node_exporter/textfile_collector/ia_alerts.prom"
TMP_FILE = PROM_FILE + ".tmp"

STATE_DIR = Path("state")
STATE_DIR.mkdir(exist_ok=True, parents=True)
EVENT_COUNTER_FILE = STATE_DIR / "event_start_counts.json"


# =========================
# HELPERS
# =========================
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
    return classify_metric(metric_name)


def sev_label_from_score(score: float) -> str:
    if score > 0.80:
        return "CRITICAL"
    if score > 0.65:
        return "HIGH"
    if score > 0.55:
        return "MEDIUM"
    return "LOW"


def sev_value(level: str) -> int:
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(level, 0)


def load_event_counter() -> dict:
    if EVENT_COUNTER_FILE.exists():
        try:
            with open(EVENT_COUNTER_FILE, "r") as f:
                data = json.load(f)
            return {k: int(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def save_event_counter(counter: dict) -> None:
    tmp = str(EVENT_COUNTER_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: int(v) for k, v in counter.items()}, f, indent=2, sort_keys=True)
    os.replace(tmp, EVENT_COUNTER_FILE)


def increment_event_counter(counter: dict, event_start_latest: dict) -> dict:
    for srv, v in event_start_latest.items():
        if int(v) == 1:
            counter[srv] = int(counter.get(srv, 0)) + 1
        else:
            counter.setdefault(srv, int(counter.get(srv, 0)))
    return counter


# =========================
# MAIN
# =========================
def main():
    final_path = OUT_DIR / "final_scores.csv"
    events_path = OUT_DIR / "events.csv"
    last_path = OUT_DIR / "alerts_last.csv"

    if not final_path.exists():
        raise FileNotFoundError(f"No existe {final_path}. Ejecuta primero predict.py")
    if not last_path.exists():
        raise FileNotFoundError(f"No existe {last_path}. Ejecuta primero predict.py")

    final = pd.read_csv(final_path)
    last_state = pd.read_csv(last_path)

    if events_path.exists():
        events_out = pd.read_csv(events_path)
    else:
        events_out = pd.DataFrame(columns=["t15", "server", "resource", "dominant_metric", "severity", "anomaly_score", "threshold"])

    # =========================
    # EVENT START LATEST
    # =========================
    event_start_latest = {}
    final["t15"] = pd.to_datetime(final["t15"], errors="coerce")

    for srv, g in final.sort_values("t15").groupby("server"):
        gg = g.tail(2).copy()
        if len(gg) < 2:
            event_start_latest[srv] = 0
            continue
        prev_s = bool(gg.iloc[0]["is_sustained"])
        curr_s = bool(gg.iloc[1]["is_sustained"])
        event_start_latest[srv] = 1 if (curr_s and (not prev_s)) else 0

    # =========================
    # EVENT START TOTAL
    # =========================
    event_counter = load_event_counter()
    event_counter = increment_event_counter(event_counter, event_start_latest)
    save_event_counter(event_counter)

    # =========================
    # EXPORT TO PROMETHEUS
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

    print("✅ Alertas exportadas correctamente")


if __name__ == "__main__":
    main()
