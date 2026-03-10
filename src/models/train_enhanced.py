import pandas as pd
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    return mae, rmse


def main():
    # -----------------------------------
    # 1. Load enhanced dataset
    # -----------------------------------
    df = pd.read_csv("data/final/enhanced_model_data.csv")

    # -----------------------------------
    # 2. Split features and target
    # -----------------------------------
    X = df.drop(columns=["task_duration_hours"])
    y = df["task_duration_hours"]

    # -----------------------------------
    # 3. Train/test split
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # 4. Define models
    # -----------------------------------
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    # -----------------------------------
    # 5. Evaluate models
    # -----------------------------------
    results = []

    for model_name, model in models.items():
        mae, rmse = evaluate_model(model, X_train, X_test, y_train, y_test)

        cv_scores = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        )

        results.append({
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std()
        })

    results_df = pd.DataFrame(results)

    # -----------------------------------
    # 6. Save metrics
    # -----------------------------------
    results_df.to_csv("results/tables/enhanced_metrics.csv", index=False)

    print("Enhanced model training completed successfully.")
    print(results_df)


if __name__ == "__main__":
    main()