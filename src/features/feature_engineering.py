import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


def main():
    # -----------------------------------
    # 1. Connect to MySQL
    # -----------------------------------
    password = quote_plus("Mirza@786")
    engine = create_engine(f"mysql+pymysql://root:{password}@localhost/workflow_db")

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
    # 5. Create time-based features
    # -----------------------------------
    df["opened_hour"] = df["opened_at"].dt.hour
    df["opened_dayofweek"] = df["opened_at"].dt.dayofweek
    df["is_weekend"] = df["opened_dayofweek"].isin([5, 6]).astype(int)

    # -----------------------------------
    # 6. Create behavior flags
    # -----------------------------------
    df["reopen_flag"] = (df["reopen_count"] > 0).astype(int)
    df["high_reassignment_flag"] = (df["reassignment_count"] > 1).astype(int)
    df["high_priority_flag"] = df["priority"].astype(str).str.lower().isin(
        ["1 - critical", "2 - high", "high", "critical"]
    ).astype(int)

    # -----------------------------------
    # 7. Group average features
    # -----------------------------------
    if "category" in df.columns:
        category_avg = df.groupby("category")["task_duration_hours"].mean().to_dict()
        df["category_avg_duration"] = df["category"].map(category_avg)

    if "assignment_group" in df.columns:
        group_avg = df.groupby("assignment_group")["task_duration_hours"].mean().to_dict()
        df["assignment_group_avg_duration"] = df["assignment_group"].map(group_avg)

    # -----------------------------------
    # 8. Save engineered dataset
    # -----------------------------------
    df.to_csv("data/processed/enhanced_feature_data.csv", index=False)

    print("Feature engineering completed successfully.")
    print(f"Rows saved: {len(df)}")


if __name__ == "__main__":
    main()