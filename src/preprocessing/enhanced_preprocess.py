import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    # -----------------------------------
    # 1. Load engineered data
    # -----------------------------------
    df = pd.read_csv("data/processed/enhanced_feature_data.csv")

    # -----------------------------------
    # 2. Fill missing categoricals
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
    # 3. Encode categoricals
    # -----------------------------------
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # -----------------------------------
    # 4. Select features
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
            "is_weekend",
            "reassignment_count",
            "reopen_count",
            "sys_mod_count",
            "reopen_flag",
            "high_reassignment_flag",
            "high_priority_flag",
            "category_avg_duration",
            "assignment_group_avg_duration"
        ] if col in df.columns
    ]

    model_df = df[feature_cols + ["task_duration_hours"]].copy()

    # -----------------------------------
    # 5. Scale numeric features
    # -----------------------------------
    numeric_cols = [
        "opened_hour",
        "opened_dayofweek",
        "reassignment_count",
        "reopen_count",
        "sys_mod_count",
        "category_avg_duration",
        "assignment_group_avg_duration"
    ]

    scaler = StandardScaler()
    existing_numeric = [c for c in numeric_cols if c in model_df.columns]

    if existing_numeric:
        model_df[existing_numeric] = scaler.fit_transform(model_df[existing_numeric])

    # -----------------------------------
    # 6. Save final enhanced model data
    # -----------------------------------
    model_df.to_csv("data/final/enhanced_model_data.csv", index=False)

    print("Enhanced preprocessing completed successfully.")
    print(f"Rows in final dataset: {len(model_df)}")
    print(f"Columns used: {model_df.columns.tolist()}")


if __name__ == "__main__":
    main()