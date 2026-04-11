import pandas as pd
from scripts.predict import build_windows, get_ip


def test_get_ip_predict():
    instance = "192.168.1.5:9182"
    assert get_ip(instance) == "192.168.1.5"


def test_build_windows_predict():
    data = pd.DataFrame({
        "t15": pd.date_range("2026-01-01", periods=3, freq="15min"),
        "instance": ["192.168.1.5:9182"] * 3,
        "ip": ["192.168.1.5"] * 3,
        "cpu_pct": [60, 62, 65],
        "mem_pct": [50, 52, 53]
    })

    result = build_windows(data, window_n=1, instance_name="192.168.1.5:9182")
    assert isinstance(result, pd.DataFrame)
    assert "ip" in result.columns
    assert not result.empty
