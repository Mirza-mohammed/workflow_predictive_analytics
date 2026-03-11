import math
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse
    }


def main():
    # -----------------------------------
    # 1. Load baseline processed dataset
    # -----------------------------------
    df = pd.read_csv("data/processed/baseline_processed.csv")

    # -----------------------------------
    # 2. Parse opened_at and sort by time
    # -----------------------------------
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["opened_at", "task_duration_hours"])
    df = df.sort_values("opened_at").reset_index(drop=True)

    # -----------------------------------
    # 3. Define safe creation-time features
    # -----------------------------------
    categorical_features = [
        col for col in [
            "priority",
            "category",
            "subcategory",
            "impact",
            "urgency",
            "contact_type",
            "location"
        ] if col in df.columns
    ]

    numeric_features = [
        col for col in [
            "opened_hour",
            "opened_dayofweek"
        ] if col in df.columns
    ]

    feature_cols = categorical_features + numeric_features

    # -----------------------------------
    # 4. Time-based split
    # -----------------------------------
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["task_duration_hours"]

    X_test = test_df[feature_cols]
    y_test = test_df["task_duration_hours"]

    # -----------------------------------
    # 5. Build preprocessing pipeline
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
    # 6. Define models
    # -----------------------------------
    dummy_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="median"))
    ])

    linear_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # -----------------------------------
    # 7. Evaluate models
    # -----------------------------------
    results = []

    results.append(
        evaluate_model(dummy_model, X_train, X_test, y_train, y_test, "DummyMedian")
    )

    results.append(
        evaluate_model(linear_model, X_train, X_test, y_train, y_test, "LinearRegression")
    )

    results_df = pd.DataFrame(results)

    # -----------------------------------
    # 8. Save metrics
    # -----------------------------------
    results_df.to_csv("results/tables/baseline_metrics.csv", index=False)

    # -----------------------------------
    # 9. Print output
    # -----------------------------------
    print("Baseline model training completed successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(results_df)


if __name__ == "__main__":
    main()