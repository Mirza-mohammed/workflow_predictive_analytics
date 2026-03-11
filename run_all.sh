#!/bin/bash
set -e

echo "Running baseline ETL..."
python -m src.etl.baseline_etl

echo "Running enhanced ETL..."
python -m src.etl.enhanced_etl

echo "Running baseline preprocessing..."
python -m src.preprocessing.baseline_preprocess

echo "Running feature engineering..."
python -m src.features.feature_engineering

echo "Running baseline training..."
python src/models/train_baseline.py

echo "Running enhanced training..."
python src/models/train_enhanced.py

echo "Running comparison..."
python src/evaluation/compare_results.py

echo "Running statistical testing..."
python src/evaluation/statistical_test.py

echo "Pipeline completed successfully."