import pandas as pd


def test_duration_not_negative():
    df = pd.DataFrame({
        "opened_at": pd.to_datetime(["2024-01-01 10:00"]),
        "resolved_at": pd.to_datetime(["2024-01-01 12:00"])
    })

    df["task_duration_hours"] = (
        (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600
    )

    assert df["task_duration_hours"].iloc[0] >= 0


def test_invalid_timestamp_becomes_missing():
    df = pd.DataFrame({
        "opened_at": ["invalid_date"]
    })

    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)

    assert df["opened_at"].isna().iloc[0]


def test_negative_duration_detected():
    df = pd.DataFrame({
        "opened_at": pd.to_datetime(["2024-01-01 12:00"]),
        "resolved_at": pd.to_datetime(["2024-01-01 10:00"])
    })

    df["task_duration_hours"] = (
        (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600
    )

    assert df["task_duration_hours"].iloc[0] < 0