import pandas as pd


def main():
    # -----------------------------------
    # 1. Load baseline and enhanced metrics
    # -----------------------------------
    baseline_df = pd.read_csv("results/tables/baseline_metrics.csv")
    enhanced_df = pd.read_csv("results/tables/enhanced_metrics.csv")

    baseline_df["system"] = "baseline"
    enhanced_df["system"] = "enhanced"

    # -----------------------------------
    # 2. Combine results
    # -----------------------------------
    combined_df = pd.concat([baseline_df, enhanced_df], ignore_index=True)

    # -----------------------------------
    # 3. Save final comparison
    # -----------------------------------
    combined_df.to_csv("results/tables/final_comparison.csv", index=False)

    print("Comparison table created successfully.")
    print(combined_df)


if __name__ == "__main__":
    main()