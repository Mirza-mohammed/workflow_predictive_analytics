import pandas as pd


def main():
    # -----------------------------------
    # 1. Load metrics
    # -----------------------------------
    baseline_df = pd.read_csv("results/tables/baseline_metrics.csv")
    enhanced_df = pd.read_csv("results/tables/enhanced_metrics.csv")

    # -----------------------------------
    # 2. Load ETL logs
    # -----------------------------------
    baseline_etl = pd.read_csv("logs/baseline_etl_log.csv")
    enhanced_etl = pd.read_csv("logs/enhanced_etl_log.csv")

    baseline_runtime = baseline_etl.loc[0, "runtime_seconds"]
    enhanced_runtime = enhanced_etl.loc[0, "runtime_seconds"]

    # -----------------------------------
    # 3. Add metadata
    # -----------------------------------
    baseline_df["system"] = "baseline"
    baseline_df["etl_runtime_seconds"] = baseline_runtime
    baseline_df["notes"] = "batch ETL + simple features"

    enhanced_df["system"] = "enhanced"
    enhanced_df["etl_runtime_seconds"] = enhanced_runtime
    enhanced_df["notes"] = "validated ETL + engineered features"

    # -----------------------------------
    # 4. Ensure baseline has matching columns
    # -----------------------------------
    if "cv_mae_mean" not in baseline_df.columns:
        baseline_df["cv_mae_mean"] = None
    if "cv_mae_std" not in baseline_df.columns:
        baseline_df["cv_mae_std"] = None

    # -----------------------------------
    # 5. Combine
    # -----------------------------------
    combined_df = pd.concat([baseline_df, enhanced_df], ignore_index=True)

    # -----------------------------------
    # 6. Reorder columns
    # -----------------------------------
    final_columns = [
        "system",
        "model",
        "mae",
        "rmse",
        "cv_mae_mean",
        "cv_mae_std",
        "etl_runtime_seconds",
        "notes"
    ]

    combined_df = combined_df[final_columns]

    # -----------------------------------
    # 7. Save final comparison
    # -----------------------------------
    combined_df.to_csv("results/tables/final_comparison.csv", index=False)

    print("Final comparison table created successfully.")
    print(combined_df)


if __name__ == "__main__":
    main()