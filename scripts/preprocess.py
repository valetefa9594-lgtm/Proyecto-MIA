import pandas as pd


def get_ip(instance: str) -> str:
    return str(instance).split(":")[0].strip()


def load_and_prep(csv_path: str, sep: str = ";") -> pd.DataFrame:
    """
    Carga el CSV original, limpia tipos y transforma a formato ancho por ventana de 15 minutos.
    """
    df = pd.read_csv(csv_path, sep=sep)

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
    """
    Garantiza continuidad temporal cada 15 minutos por servidor.
    """
    full_data = []

    for srv in wide["instance"].unique():
        temp = wide[wide["instance"] == srv].copy()
        full_range = pd.date_range(temp["t15"].min(), temp["t15"].max(), freq="15min")
        grid = pd.DataFrame({"t15": full_range, "instance": srv})
        temp_full = grid.merge(temp, on=["t15", "instance"], how="left")
        temp_full["ip"] = get_ip(srv)
        full_data.append(temp_full)

    wide2 = pd.concat(full_data, ignore_index=True)
    wide2 = wide2.sort_values(["instance", "t15"]).reset_index(drop=True)

    return wide2


def impute_data(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes de forma simple y consistente:
    - discos inexistentes -> 0
    - cpu/mem -> forward fill por servidor
    - cpu/mem inicial -> mediana por servidor
    - resto -> mediana global
    """
    id_cols = ["t15", "instance", "ip"]
    feature_cols = [c for c in wide.columns if c not in id_cols]

    for col in feature_cols:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    disk_cols = [c for c in feature_cols if "disk" in c.lower()]
    if disk_cols:
        wide[disk_cols] = wide[disk_cols].fillna(0)

    for col in feature_cols:
        if ("cpu" in col.lower()) or ("mem" in col.lower()) or ("memory" in col.lower()):
            wide[col] = wide.groupby("instance")[col].ffill()

    for col in feature_cols:
        if ("cpu" in col.lower()) or ("mem" in col.lower()) or ("memory" in col.lower()):
            wide[col] = wide.groupby("instance")[col].transform(lambda x: x.fillna(x.median()))

    wide[feature_cols] = wide[feature_cols].fillna(wide[feature_cols].median(numeric_only=True))

    return wide


def preprocess_pipeline(csv_path: str, sep: str = ";") -> pd.DataFrame:
    """
    Ejecuta todo el flujo de preprocesamiento.
    """
    wide = load_and_prep(csv_path, sep=sep)
    wide = enforce_continuity(wide)
    wide = impute_data(wide)
    return wide
