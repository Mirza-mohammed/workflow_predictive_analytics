import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from src.common.config import get_db_url


def main():
    # -----------------------------------
    # 1. Start timer
    # -----------------------------------
    start_time = datetime.now()

    # -----------------------------------
    # 2. Read raw CSV
    # -----------------------------------
    file_path = "data/raw/workflow_tasks.csv"
    df = pd.read_csv(file_path)

    # -----------------------------------
    # 3. Standardize column names
    # -----------------------------------
    df.columns = [col.strip().lower() for col in df.columns]

    # -----------------------------------
    # 4. Parse timestamps
    # -----------------------------------
    timestamp_cols = [
        "opened_at",
        "sys_created_at",
        "sys_updated_at",
        "resolved_at",
        "closed_at"
    ]

    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # -----------------------------------
    # 5. Create validation flags
    # -----------------------------------
    df["valid_number"] = df["number"].notna()
    df["valid_opened_at"] = df["opened_at"].notna()
    df["valid_resolved_at"] = df["resolved_at"].notna()
    df["valid_priority"] = df["priority"].notna()
    df["valid_category"] = df["category"].notna()
    df["valid_timestamp_order"] = df["resolved_at"] >= df["opened_at"]

    # -----------------------------------
    # 6. Create duration
    # -----------------------------------
    df["task_duration_hours"] = (
        (df["resolved_at"] - df["opened_at"]).dt.total_seconds() / 3600
    )

    df["valid_duration"] = df["task_duration_hours"] >= 0

    # -----------------------------------
    # 7. Create overall validity flag
    # -----------------------------------
    validity_cols = [
        "valid_number",
        "valid_opened_at",
        "valid_resolved_at",
        "valid_priority",
        "valid_category",
        "valid_timestamp_order",
        "valid_duration"
    ]

    df["is_valid"] = df[validity_cols].all(axis=1)

    # -----------------------------------
    # 8. Split valid and invalid rows
    # -----------------------------------
    valid_df = df[df["is_valid"]].copy()
    invalid_df = df[~df["is_valid"]].copy()

    # -----------------------------------
    # 9. Save invalid rows
    # -----------------------------------
    invalid_df.to_csv("logs/invalid_rows.csv", index=False)

    # -----------------------------------
    # 10. Connect to MySQL
    # -----------------------------------
    engine = create_engine(get_db_url())

    # -----------------------------------
    # 11. Load valid rows into MySQL
    # -----------------------------------
    valid_df.to_sql(
        "workflow_tasks_enhanced",
        con=engine,
        if_exists="replace",
        index=False
    )

    # -----------------------------------
    # 12. End timer
    # -----------------------------------
    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    # -----------------------------------
    # 13. Create ETL log
    # -----------------------------------
    log_df = pd.DataFrame([{
        "pipeline_name": "enhanced_etl",
        "source_file": file_path,
        "input_rows": len(df),
        "valid_rows": len(valid_df),
        "invalid_rows": len(invalid_df),
        "columns_loaded": len(valid_df.columns),
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": runtime_seconds
    }])

    log_df.to_csv("logs/enhanced_etl_log.csv", index=False)

    print("Enhanced ETL completed successfully.")
    print(f"Input rows: {len(df)}")
    print(f"Valid rows: {len(valid_df)}")
    print(f"Invalid rows: {len(invalid_df)}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")


if __name__ == "__main__":
    main()