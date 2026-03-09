import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from urllib.parse import quote_plus


def main():
    # -----------------------------------
    # 1. Safe MySQL connection
    # -----------------------------------
    password = quote_plus("Mirza@786")
    engine = create_engine(f"mysql+pymysql://root:{password}@localhost/workflow_db")

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
    # 7. Fill missing values in categoricals
    # -----------------------------------
    categorical_cols = [
        "priority",
        "category",
        "assigned_to",
        "assignment_group",
        "incident_state"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # -----------------------------------
    # 8. Label encode categorical columns
    # -----------------------------------
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # -----------------------------------
    # 9. Choose baseline features
    # -----------------------------------
    feature_cols = [
        col for col in [
            "priority",
            "category",
            "assigned_to",
            "assignment_group",
            "incident_state",
            "opened_hour",
            "opened_dayofweek",
            "reassignment_count",
            "reopen_count",
            "sys_mod_count"
        ] if col in df.columns
    ]

    model_df = df[feature_cols + ["task_duration_hours"]].copy()

    # -----------------------------------
    # 10. Scale numeric columns
    # -----------------------------------
    numeric_cols = [
        "opened_hour",
        "opened_dayofweek",
        "reassignment_count",
        "reopen_count",
        "sys_mod_count"
    ]

    scaler = StandardScaler()
    existing_numeric = [c for c in numeric_cols if c in model_df.columns]

    if existing_numeric:
        model_df[existing_numeric] = scaler.fit_transform(model_df[existing_numeric])

    # -----------------------------------
    # 11. Save processed baseline dataset
    # -----------------------------------
    model_df.to_csv("data/processed/baseline_processed.csv", index=False)

    print("Baseline preprocessing completed successfully.")
    print(f"Rows in processed dataset: {len(model_df)}")
    print(f"Columns used: {model_df.columns.tolist()}")


if __name__ == "__main__":
    main()