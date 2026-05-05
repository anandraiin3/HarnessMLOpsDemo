"""
train_model.py — Train RandomForest classifier and save artifacts locally.

Pure ML step — no AWS dependencies. AWS integrations are handled as
separate, named steps within the same Harness pipeline stage:
  Step 2: log_experiment.py    → SageMaker Experiments
  Step 3: upload_artifacts.py  → S3 artifact storage
  Step 4: register_model.py    → SageMaker Model Registry

Usage:
    python scripts/train_model.py --n_estimators 100 --model_name credit-card-approval
"""

import argparse
import json
import os
import yaml
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ── Metric helpers ─────────────────────────────────────────────────────────────

def get_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall':    recall_score(y_true, y_pred, average='weighted'),
        'f1':        f1_score(y_true, y_pred, average='weighted'),
        'accuracy':  accuracy_score(y_true, y_pred),
    }


def get_fairness_metrics(y_pred, group_test):
    """Compute per-group approval rates and the approval-rate fairness gap."""
    metrics = {}
    approval_rates = []
    for grp_val in sorted(group_test.unique()):
        mask = group_test.values == grp_val
        rate = float(pd.Series(y_pred)[mask].mean())
        metrics[f"fairness_group_{grp_val}_approval_rate"] = rate
        approval_rates.append(rate)
    metrics["fairness_gap"] = (
        max(approval_rates) - min(approval_rates)
        if len(approval_rates) >= 2 else 0.0
    )
    return metrics


def get_feature_importances(X, y):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)
    return pd.Series(forest.feature_importances_, index=X.columns).to_dict()


# ── Training ───────────────────────────────────────────────────────────────────

def train(X_train, X_test, y_train, y_test, group_test, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_metrics = get_metrics(y_train, y_train_pred)
    test_metrics  = get_metrics(y_test,  y_test_pred)
    fairness      = get_fairness_metrics(y_test_pred, group_test)
    importances   = get_feature_importances(X_train, y_train)

    return model, train_metrics, test_metrics, fairness, importances


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RandomForest classifier.')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='credit-card-approval')
    args = parser.parse_args()

    with open("configs/config.yml") as f:
        config = yaml.safe_load(f)

    run_name = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"\n==> Starting training run: {run_name}")

    # ── Load data ──────────────────────────────────────────────────────────────
    data = pd.read_csv(config['data']['load_file_path'])
    group = data["Group"] if "Group" in data.columns else pd.Series(
        ["unknown"] * len(data), name="Group"
    )
    X = data.drop(columns=["Target", "Group"], errors='ignore')
    y = data["Target"]

    X_train, X_test, y_train, y_test, _, group_test = train_test_split(
        X, y, group, test_size=0.2, random_state=42
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    model, train_metrics, test_metrics, fairness, importances = train(
        X_train, X_test, y_train, y_test, group_test, args.n_estimators
    )

    all_metrics = {
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **fairness,
    }
    params = {
        "n_estimators":        args.n_estimators,
        "model_name":          args.model_name,
        "n_features":          X.shape[1],
        "feature_names":       list(X.columns),
        "feature_importances": importances,
    }

    # ── Save artifacts locally ─────────────────────────────────────────────────
    # Shared workspace: subsequent steps in this stage read these files.
    # S3 upload happens in the next step (upload_artifacts.py).
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open("outputs/params.json", "w") as f:
        json.dump(params, f, indent=2)

    joblib.dump(model, "outputs/model.joblib")

    with open("outputs/run_name.txt", "w") as f:
        f.write(run_name)

    print(f"\n==> Run name       : {run_name}")
    print(f"==> Train accuracy  : {train_metrics['accuracy']:.4f}")
    print(f"==> Test  accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"==> Fairness gap    : {fairness['fairness_gap']:.4f}")
    print(f"\nArtifacts saved to outputs/ — next step: Log to SageMaker Experiments")


if __name__ == "__main__":
    main()
