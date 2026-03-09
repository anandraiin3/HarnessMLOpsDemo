import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def get_metrics(y_true, y_pred):
    """
    Compute and return basic classification metrics, adding a small random
    noise to each metric.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        metrics (dict): Contains confusion matrix, precision, recall, F1, accuracy.
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, average='weighted') + np.random.uniform(-0.09, 0.1),
        'recall': recall_score(y_true, y_pred, average='weighted') + np.random.uniform(-0.09, 0.1),
        'f1': f1_score(y_true, y_pred, average='weighted') + np.random.uniform(-0.09, 0.1),
        'accuracy': accuracy_score(y_true, y_pred) + np.random.uniform(-0.09, 0.1)
    }
    return metrics


def get_feature_importances(X, y):
    """
    Train a RandomForest on the dataset to compute and plot
    feature importances (Mean Decrease in Impurity).

    Args:
        X (DataFrame): Features.
        y (Series): Labels.

    Returns:
        (str): JSON-encoded dictionary of feature importances.
    """
    feature_names = X.columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)

    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)

    return forest_importances.to_json()


def train_random_forest(n_estimators):
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Convert X to DataFrame for easier column access
    X = pd.DataFrame(X, columns=data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Predictions - Unawareness
        y_train_unaware_pred = model.predict(X_train)
        y_test_unaware_pred = model.predict(X_test)

        # Metrics - Unawareness
        train_metrics_unaware = get_metrics(y_train, y_train_unaware_pred)
        test_metrics_unaware = get_metrics(y_test, y_test_unaware_pred)
        feature_importances_unaware = get_feature_importances(X_train, y_train)

        # Save selected model & metrics
        selected_model_metrics = {
            'train': train_metrics_unaware,
            'test': test_metrics_unaware,
            'feature_importances': feature_importances_unaware
        }

        # Log parameters, metrics, and model
        mlflow.log_params({
            "n_estimators": n_estimators,
            "n_features": X.shape[1],
            "feature_importances": feature_importances_unaware
        })

        for split in ["train", "test"]:
            for metric, value in selected_model_metrics[split].items():
                mlflow.log_metric(f"{split}_{metric}", value)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Random Forest model.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest.')
    args = parser.parse_args()

    train_random_forest(n_estimators=args.n_estimators)
