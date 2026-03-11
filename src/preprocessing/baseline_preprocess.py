import pandas as pd
from sqlalchemy import create_engine
from src.common.config import get_db_url


def main():
    # -----------------------------------
    # 1. Connect to MySQL
    # -----------------------------------
    engine = create_engine(get_db_url())

    # -----------------------------------
    # 2. Read data from MySQL
    # -----------------------------------
    df = pd.read_sql("SELECT * FROM workflow_tasks", con=engine)

    # -----------------------------------
    # 3. Parse timestamps
    # -----------------------------------
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", dayfirst=True)

    # -----------------------------------
    # 4. Create target variable
    # -----------------------------------
    df["task_duration_hours"] = (
        (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600
    )

    # -----------------------------------
    # 5. Drop rows where target is missing
    # -----------------------------------
    df = df.dropna(subset=["task_duration_hours"])

    # -----------------------------------
    # 6. Create simple temporal features
    # -----------------------------------
    df["opened_hour"] = df["opened_at"].dt.hour
    df["opened_dayofweek"] = df["opened_at"].dt.dayofweek

    # -----------------------------------
    # 7. Keep a raw baseline-safe dataset
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
            "opened_dayofweek"
        ] if col in df.columns
    ]

    model_df = df[safe_feature_cols + ["task_duration_hours"]].copy()

    # -----------------------------------
    # 8. Save baseline dataset for training
    # -----------------------------------
    model_df.to_csv("data/processed/baseline_processed.csv", index=False)

    print("Baseline preprocessing completed successfully.")
    print(f"Rows in processed dataset: {len(model_df)}")
    print(f"Columns used: {model_df.columns.tolist()}")


if __name__ == "__main__":
    main()