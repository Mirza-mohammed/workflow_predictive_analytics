import pandas as pd
from scipy.stats import wilcoxon


def compare_models(cv_df, model_a, model_b):
    a_scores = cv_df[cv_df["model"] == model_a].sort_values("fold")["mae"].values
    b_scores = cv_df[cv_df["model"] == model_b].sort_values("fold")["mae"].values

    if len(a_scores) != len(b_scores):
        raise ValueError(f"Fold count mismatch between {model_a} and {model_b}")

    stat, p_value = wilcoxon(a_scores, b_scores)

    mean_a = a_scores.mean()
    mean_b = b_scores.mean()

    if mean_b < mean_a:
        better_model = model_b
    elif mean_a < mean_b:
        better_model = model_a
    else:
        better_model = "tie"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_a_mean_mae": mean_a,
        "model_b_mean_mae": mean_b,
        "better_model": better_model,
        "wilcoxon_statistic": stat,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05
    }


def main():
    # -----------------------------------
    # 1. Load fold-level CV scores
    # -----------------------------------
    cv_df = pd.read_csv("results/tables/enhanced_cv_scores.csv")

    # -----------------------------------
    # 2. Define comparisons
    # -----------------------------------
    comparisons = [
        ("DummyMedian", "RandomForest"),
        ("DummyMedian", "GradientBoosting"),
        ("RandomForest", "GradientBoosting")
    ]

    # -----------------------------------
    # 3. Run statistical tests
    # -----------------------------------
    results = []

    for model_a, model_b in comparisons:
        result = compare_models(cv_df, model_a, model_b)
        results.append(result)

    results_df = pd.DataFrame(results)

    # -----------------------------------
    # 4. Save results
    # -----------------------------------
    results_df.to_csv("results/tables/statistical_test_results.csv", index=False)

    print("Statistical testing completed successfully.")
    print(results_df)


if __name__ == "__main__":
    main()