import pandas as pd
import os


def test_baseline_metrics_file_exists():
    assert os.path.exists("results/tables/baseline_metrics.csv")


def test_enhanced_metrics_file_exists():
    assert os.path.exists("results/tables/enhanced_metrics.csv")


def test_final_comparison_file_exists():
    assert os.path.exists("results/tables/final_comparison.csv")


def test_enhanced_cv_scores_file_exists():
    assert os.path.exists("results/tables/enhanced_cv_scores.csv")


def test_baseline_contains_dummy_model():
    df = pd.read_csv("results/tables/baseline_metrics.csv")
    assert "DummyMedian" in df["model"].values


def test_baseline_contains_linear_regression():
    df = pd.read_csv("results/tables/baseline_metrics.csv")
    assert "LinearRegression" in df["model"].values


def test_enhanced_contains_dummy_model():
    df = pd.read_csv("results/tables/enhanced_metrics.csv")
    assert "DummyMedian" in df["model"].values


def test_enhanced_contains_random_forest():
    df = pd.read_csv("results/tables/enhanced_metrics.csv")
    assert "RandomForest" in df["model"].values


def test_enhanced_contains_gradient_boosting():
    df = pd.read_csv("results/tables/enhanced_metrics.csv")
    assert "GradientBoosting" in df["model"].values