import pandas as pd


def test_no_target_leaking_average_features():
    df = pd.read_csv("data/processed/enhanced_feature_data.csv")

    assert "category_avg_duration" not in df.columns
    assert "assignment_group_avg_duration" not in df.columns


def test_safe_feature_columns_exist():
    df = pd.read_csv("data/processed/enhanced_feature_data.csv")

    expected_columns = [
        "opened_at",
        "priority",
        "category",
        "subcategory",
        "impact",
        "urgency",
        "contact_type",
        "location",
        "opened_hour",
        "opened_dayofweek",
        "is_weekend",
        "high_priority_flag",
        "task_duration_hours"
    ]

    for col in expected_columns:
        assert col in df.columns