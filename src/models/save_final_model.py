import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


def main():
    # -----------------------------------
    # 1. Load final enhanced dataset
    # -----------------------------------
    df = pd.read_csv("data/processed/enhanced_feature_data.csv")

    # -----------------------------------
    # 2. Parse and sort by time
    # -----------------------------------
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["opened_at", "task_duration_hours"])
    df = df.sort_values("opened_at").reset_index(drop=True)

    # -----------------------------------
    # 3. Define final safe feature set
    # -----------------------------------
    categorical_features = [
        col for col in [
            "priority",
            "category",
            "subcategory",
            "impact",
            "urgency",
            "contact_type",
            "location",
            "u_symptom",
            "cmdb_ci"
        ] if col in df.columns
    ]

    numeric_features = [
        col for col in [
            "opened_hour",
            "opened_dayofweek",
            "is_weekend",
            "high_priority_flag"
        ] if col in df.columns
    ]

    feature_cols = categorical_features + numeric_features

    X = df[feature_cols]
    y = df["task_duration_hours"]

    # -----------------------------------
    # 4. Build preprocessing
    # -----------------------------------
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ]
    )

    # -----------------------------------
    # 5. Build final model
    # -----------------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    # -----------------------------------
    # 6. Train on log-transformed target
    # -----------------------------------
    y_log = np.log1p(y)
    model.fit(X, y_log)

    artifact = {
        "model": model,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "feature_cols": feature_cols
    }

    # -----------------------------------
    # 7. Save model artifact
    # -----------------------------------
    joblib.dump(artifact, "models/final_gradient_boosting_pipeline.joblib")
    print("Saved model to models/final_gradient_boosting_pipeline.joblib")


if __name__ == "__main__":
    main()