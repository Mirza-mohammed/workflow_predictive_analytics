import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from urllib.parse import quote_plus


def main():
    # ---------------------------
    # 1. Start timer
    # ---------------------------
    start_time = datetime.now()

    # ---------------------------
    # 2. Read raw CSV file
    # ---------------------------
    file_path = "data/raw/workflow_tasks.csv"
    df = pd.read_csv(file_path)

    # ---------------------------
    # 3. Standardize column names
    # ---------------------------
    df.columns = [col.strip().lower() for col in df.columns]

    # ---------------------------
    # 4. Convert timestamp columns
    # ---------------------------
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

    # ---------------------------
    # 5. Connect to MySQL
    # ---------------------------
    password = quote_plus("Mirza@786")
    engine = create_engine(
        f"mysql+pymysql://root:{password}@localhost/workflow_db"
    )

    # ---------------------------
    # 6. Load data into MySQL
    # ---------------------------
    df.to_sql("workflow_tasks", con=engine, if_exists="replace", index=False)

    # ---------------------------
    # 7. End timer
    # ---------------------------
    end_time = datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    # ---------------------------
    # 8. Create ETL log
    # ---------------------------
    log_df = pd.DataFrame([{
        "pipeline_name": "baseline_etl",
        "source_file": file_path,
        "rows_loaded": len(df),
        "columns_loaded": len(df.columns),
        "start_time": start_time,
        "end_time": end_time,
        "runtime_seconds": runtime_seconds
    }])

    log_df.to_csv("logs/baseline_etl_log.csv", index=False)

    print("Baseline ETL completed successfully.")
    print(f"Rows loaded: {len(df)}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")


if __name__ == "__main__":
    main()