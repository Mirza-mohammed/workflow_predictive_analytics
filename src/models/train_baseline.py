import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # -----------------------------------
    # 1. Load processed dataset
    # -----------------------------------
    df = pd.read_csv("data/processed/baseline_processed.csv")

    # -----------------------------------
    # 2. Separate features and target
    # -----------------------------------
    X = df.drop(columns=["task_duration_hours"])
    y = df["task_duration_hours"]

    # -----------------------------------
    # 3. Split into train and test sets
    # -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # 4. Create and train baseline model
    # -----------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -----------------------------------
    # 5. Make predictions
    # -----------------------------------
    predictions = model.predict(X_test)

    # -----------------------------------
    # 6. Evaluate the model
    # -----------------------------------
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))

    # -----------------------------------
    # 7. Save metrics
    # -----------------------------------
    results_df = pd.DataFrame([{
        "model": "LinearRegression",
        "mae": mae,
        "rmse": rmse
    }])

    results_df.to_csv("results/tables/baseline_metrics.csv", index=False)

    # -----------------------------------
    # 8. Print output
    # -----------------------------------
    print("Baseline model training completed successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()