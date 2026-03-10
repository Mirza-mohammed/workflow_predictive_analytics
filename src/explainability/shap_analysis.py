import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def main():
    # -----------------------------------
    # 1. Load enhanced final dataset
    # -----------------------------------
    df = pd.read_csv("data/final/enhanced_model_data.csv")

    # -----------------------------------
    # 2. Separate features and target
    # -----------------------------------
    X = df.drop(columns=["task_duration_hours"])
    y = df["task_duration_hours"]

    # -----------------------------------
    # 3. Train the chosen enhanced model
    # -----------------------------------
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # -----------------------------------
    # 4. Use a smaller sample for SHAP
    # -----------------------------------
    X_sample = X.sample(n=min(500, len(X)), random_state=42)

    # -----------------------------------
    # 5. Create SHAP explainer
    # -----------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # -----------------------------------
    # 6. Save SHAP summary plot
    # -----------------------------------
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("results/figures/shap_summary.png")
    plt.close()

    print("SHAP analysis completed successfully.")
    print("Saved: results/figures/shap_summary.png")


if __name__ == "__main__":
    main()