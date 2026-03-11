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
    baseline_df["notes"] = baseline_df["model"].apply(
        lambda m: "naive median baseline" if m == "DummyMedian" else "batch ETL + safe creation-time features"
    )

    enhanced_df["system"] = "enhanced"
    enhanced_df["etl_runtime_seconds"] = enhanced_runtime
    enhanced_df["notes"] = enhanced_df["model"].apply(
        lambda m: "naive median baseline" if m == "DummyMedian" else "validated ETL + engineered safe features"
    )

    # -----------------------------------
    # 4. Ensure matching columns
    # -----------------------------------
    for df_ in [baseline_df, enhanced_df]:
        if "cv_mae_mean" not in df_.columns:
            df_["cv_mae_mean"] = None
        if "cv_mae_std" not in df_.columns:
            df_["cv_mae_std"] = None

    # -----------------------------------
    # 5. Combine results
    # -----------------------------------
    combined_df = pd.concat([baseline_df, enhanced_df], ignore_index=True)

    # -----------------------------------
    # 6. Add improvement vs dummy baseline
    # -----------------------------------
    dummy_rows = combined_df[combined_df["model"] == "DummyMedian"]

    baseline_dummy_mae = None
    enhanced_dummy_mae = None

    if not dummy_rows.empty:
        baseline_dummy = dummy_rows[dummy_rows["system"] == "baseline"]
        enhanced_dummy = dummy_rows[dummy_rows["system"] == "enhanced"]

        if not baseline_dummy.empty:
            baseline_dummy_mae = baseline_dummy.iloc[0]["mae"]
        if not enhanced_dummy.empty:
            enhanced_dummy_mae = enhanced_dummy.iloc[0]["mae"]

    def calculate_improvement(row):
        if row["model"] == "DummyMedian":
            return None

        if row["system"] == "baseline" and baseline_dummy_mae:
            return ((baseline_dummy_mae - row["mae"]) / baseline_dummy_mae) * 100

        if row["system"] == "enhanced" and enhanced_dummy_mae:
            return ((enhanced_dummy_mae - row["mae"]) / enhanced_dummy_mae) * 100

        return None

    combined_df["mae_improvement_vs_dummy_pct"] = combined_df.apply(calculate_improvement, axis=1)

    # -----------------------------------
    # 7. Add pass/fail against 20% threshold
    # -----------------------------------
    def pass_fail(row):
        if row["model"] == "DummyMedian":
            return "baseline_reference"

        improvement = row["mae_improvement_vs_dummy_pct"]
        if improvement is None:
            return "unknown"

        return "pass" if improvement >= 20 else "fail"

    combined_df["meets_success_threshold"] = combined_df.apply(pass_fail, axis=1)

    # -----------------------------------
    # 8. Reorder columns
    # -----------------------------------
    final_columns = [
        "system",
        "model",
        "mae",
        "rmse",
        "cv_mae_mean",
        "cv_mae_std",
        "etl_runtime_seconds",
        "mae_improvement_vs_dummy_pct",
        "meets_success_threshold",
        "notes"
    ]

    combined_df = combined_df[final_columns]

    # -----------------------------------
    # 9. Save final comparison
    # -----------------------------------
    combined_df.to_csv("results/tables/final_comparison.csv", index=False)

    print("Final comparison table created successfully.")
    print(combined_df)


if __name__ == "__main__":
    main()