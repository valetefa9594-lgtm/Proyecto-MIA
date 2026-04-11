import pandas as pd
from scripts.preprocess import get_ip, load_and_prep


def test_get_ip():
    instance = "192.168.1.2:9182"
    assert get_ip(instance) == "192.168.1.2"


def test_load_and_prep_returns_dataframe():
    df = load_and_prep("data/sample/ejemplo_dataset.csv", sep=";")
    assert isinstance(df, pd.DataFrame)
    assert "t15" in df.columns
    assert "instance" in df.columns
