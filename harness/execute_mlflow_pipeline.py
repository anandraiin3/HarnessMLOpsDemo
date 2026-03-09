import argparse
import os
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
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
    Compute and return basic classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        metrics (dict): Contains precision, recall, F1, accuracy.
    """
    return {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall':    recall_score(y_true, y_pred, average='weighted'),
        'f1':        f1_score(y_true, y_pred, average='weighted'),
        'accuracy':  accuracy_score(y_true, y_pred),
    }


def get_feature_importances(X, y):
    """
    Train a RandomForest to compute feature importances (Mean Decrease Impurity).

    Args:
        X (DataFrame): Features.
        y (Series):    Labels.

    Returns:
        (str): JSON-encoded dictionary of feature importances.
    """
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)
    return pd.Series(forest.feature_importances_, index=X.columns).to_json()


def train_random_forest(X, y, n_estimators, model_name):
    """
    Train a RandomForestClassifier, log everything to MLFlow, register the model.

    Args:
        X (DataFrame): Features (Group column already dropped).
        y (Series):    Labels.
        n_estimators (int): Number of trees.
        model_name (str): MLFlow registered model name.

    Returns:
        run_id (str): The MLFlow run ID.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_metrics = get_metrics(y_train, y_train_pred)
        test_metrics  = get_metrics(y_test,  y_test_pred)
        feature_importances = get_feature_importances(X_train, y_train)

        # Log parameters
        mlflow.log_params({
            "n_estimators":      n_estimators,
            "n_features":        X.shape[1],
            "feature_names":     list(X.columns),
            "feature_importances": feature_importances,
        })

        # Log metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Log and register model
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        run_id = run.info.run_id

    print(f"\n==> MLFlow Run ID : {run_id}")
    print(f"==> Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"==> Test  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"==> Model registered as '{model_name}'")

    return run_id


def main():
    parser = argparse.ArgumentParser(description='Train a Random Forest model.')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the forest.')
    parser.add_argument('--model_name', type=str, default='credit-card-approval',
                        help='MLFlow registered model name.')
    args = parser.parse_args()

    config_file_path = "config.yml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    tracking_uri = config.get('mlflow', {}).get(
        'tracking_uri', os.environ.get('MLFLOW_TRACKING_URI', '')
    )
    experiment_name = config.get('mlflow', {}).get(
        'experiment_name', 'Credit Card Approval'
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data = pd.read_csv(config['data']['load_file_path'])

    # Drop target and sensitive Group attribute (fairness-unaware model)
    X = data.drop(columns=["Target", "Group"], errors='ignore')
    y = data["Target"]

    run_id = train_random_forest(X, y, args.n_estimators, args.model_name)

    # Write run ID to file for downstream pipeline steps
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/run_id.txt", "w") as f:
        f.write(run_id)
    print(f"==> Run ID written to outputs/run_id.txt")


if __name__ == "__main__":
    main()
