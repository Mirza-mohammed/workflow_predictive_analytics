import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, use_log_target=False):
    if use_log_target:
        y_train_transformed = np.log1p(y_train)
        model.fit(X_train, y_train_transformed)

        predictions_log = model.predict(X_test)
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse
    }


def evaluate_time_series_cv(model, X, y, model_name, use_log_target=False, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_index, valid_index) in enumerate(tscv.split(X), start=1):
        X_train_fold = X.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]

        y_train_fold = y.iloc[train_index]
        y_valid_fold = y.iloc[valid_index]

        model_fold = clone(model)

        if use_log_target:
            y_train_fold_transformed = np.log1p(y_train_fold)
            model_fold.fit(X_train_fold, y_train_fold_transformed)

            predictions_log = model_fold.predict(X_valid_fold)
            predictions = np.expm1(predictions_log)
            predictions = np.maximum(predictions, 0)
        else:
            model_fold.fit(X_train_fold, y_train_fold)
            predictions = model_fold.predict(X_valid_fold)

        mae = mean_absolute_error(y_valid_fold, predictions)

        fold_results.append({
            "model": model_name,
            "fold": fold_idx,
            "mae": mae
        })

    return pd.DataFrame(fold_results)


def main():
    # -----------------------------------
    # 1. Load enhanced dataset
    # -----------------------------------
    df = pd.read_csv("data/processed/enhanced_feature_data.csv")

    # -----------------------------------
    # 2. Parse opened_at and sort by time
    # -----------------------------------
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["opened_at", "task_duration_hours"])
    df = df.sort_values("opened_at").reset_index(drop=True)

    # -----------------------------------
    # 2b. Outlier cap experiment
    # -----------------------------------
    cap_value = df["task_duration_hours"].quantile(0.99)
    df["task_duration_hours"] = df["task_duration_hours"].clip(upper=cap_value)

    print(f"Applied 99th percentile cap to target at: {cap_value:.2f} hours")

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

    # -----------------------------------
    # 4. Time-based train/test split
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
    models = {
        "DummyMedian": {
            "pipeline": Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", DummyRegressor(strategy="median"))
            ]),
            "use_log_target": False
        },
        "RandomForest": {
            "pipeline": Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "use_log_target": True
        },
        "GradientBoosting": {
            "pipeline": Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ))
            ]),
            "use_log_target": True
        }
    }

    # -----------------------------------
    # 7. Evaluate models
    # -----------------------------------
    results = []
    all_cv_folds = []

    X_cv = train_df[feature_cols]
    y_cv = train_df["task_duration_hours"]

    for model_name, model_info in models.items():
        model = model_info["pipeline"]
        use_log_target = model_info["use_log_target"]

        result = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name, use_log_target=use_log_target
        )

        cv_fold_df = evaluate_time_series_cv(
            model,
            X_cv,
            y_cv,
            model_name,
            use_log_target=use_log_target,
            n_splits=5
        )

        result["cv_mae_mean"] = cv_fold_df["mae"].mean()
        result["cv_mae_std"] = cv_fold_df["mae"].std()

        results.append(result)
        all_cv_folds.append(cv_fold_df)

    results_df = pd.DataFrame(results)
    cv_scores_df = pd.concat(all_cv_folds, ignore_index=True)

    # -----------------------------------
    # 8. Save metrics and fold scores
    # -----------------------------------
    results_df.to_csv("results/tables/enhanced_metrics.csv", index=False)
    cv_scores_df.to_csv("results/tables/enhanced_cv_scores.csv", index=False)

    print("Enhanced outlier-test model training completed successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(results_df)
    print("Saved fold-level CV scores to results/tables/enhanced_cv_scores.csv")


if __name__ == "__main__":
    main()