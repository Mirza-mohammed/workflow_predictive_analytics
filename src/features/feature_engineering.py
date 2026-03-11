import pandas as pd
from sqlalchemy import create_engine
from src.common.config import get_db_url


def main():
    # -----------------------------------
    # 1. Connect to MySQL
    # -----------------------------------
    engine = create_engine(get_db_url())

    # -----------------------------------
    # 2. Load enhanced ETL data
    # -----------------------------------
    df = pd.read_sql("SELECT * FROM workflow_tasks_enhanced", con=engine)

    # -----------------------------------
    # 3. Parse timestamps
    # -----------------------------------
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", dayfirst=True)

    # -----------------------------------
    # 4. Recreate duration
    # -----------------------------------
    df["task_duration_hours"] = (
        (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600
    )

    # -----------------------------------
    # 5. Create safe time-based features
    # -----------------------------------
    df["opened_hour"] = df["opened_at"].dt.hour
    df["opened_dayofweek"] = df["opened_at"].dt.dayofweek
    df["is_weekend"] = df["opened_dayofweek"].isin([5, 6]).astype(int)

    # -----------------------------------
    # 6. Create safe priority-based flag
    # -----------------------------------
    df["high_priority_flag"] = df["priority"].astype(str).str.lower().isin(
        ["1 - critical", "2 - high", "high", "critical"]
    ).astype(int)

    # -----------------------------------
    # 7. Keep only creation-time-safe fields
    # -----------------------------------
    safe_feature_cols = [
        col for col in [
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
            "high_priority_flag"
        ] if col in df.columns
    ]

    feature_df = df[safe_feature_cols + ["task_duration_hours"]].copy()

    # -----------------------------------
    # 8. Save engineered dataset
    # -----------------------------------
    feature_df.to_csv("data/processed/enhanced_feature_data.csv", index=False)

    print("Feature engineering completed successfully.")
    print(f"Rows saved: {len(feature_df)}")
    print(f"Columns saved: {feature_df.columns.tolist()}")


if __name__ == "__main__":
    main()