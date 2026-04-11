import pandas as pd
from scripts.train import build_windows


def test_build_windows_returns_dataframe():
    data = pd.DataFrame({
        "t15": pd.date_range("2026-01-01", periods=3, freq="15min"),
        "instance": ["192.168.1.2:9182"] * 3,
        "cpu_pct": [80, 82, 85],
        "mem_pct": [70, 71, 72]
    })

    result = build_windows(data, window_n=1, instance_name="192.168.1.2:9182")
    assert isinstance(result, pd.DataFrame)
    assert "instance" in result.columns
    assert not result.empty
