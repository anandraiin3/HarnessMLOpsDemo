"""
evaluate_model.py — MLFlow model evaluation gate for the Harness MLOps pipeline.

Loads metrics from a completed MLFlow run, asserts quality and fairness thresholds,
and exits with a non-zero status code if any threshold is violated.

Usage:
    python harness/evaluate_model.py --run_id <MLFLOW_RUN_ID>
    python harness/evaluate_model.py --run_id <MLFLOW_RUN_ID> --tracking_uri https://...
"""

import argparse
import os
import sys
import yaml
import mlflow


# ── Thresholds ────────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD     = 0.80   # Minimum test accuracy
PRECISION_THRESHOLD    = 0.75   # Minimum test precision
RECALL_THRESHOLD       = 0.75   # Minimum test recall
F1_THRESHOLD           = 0.75   # Minimum test F1

# Fairness: maximum allowed gap between best-group and worst-group accuracy.
# Set to None to skip fairness check (when group metrics are not logged).
FAIRNESS_ACCURACY_GAP  = None   # Reserved for future fairness logging


def load_config():
    try:
        with open("config.yml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def evaluate(run_id: str, tracking_uri: str):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    print(f"\n{'='*60}")
    print(f"  MLOps Evaluation Gate — Run: {run_id}")
    print(f"{'='*60}")

    violations = []

    checks = [
        ("test_accuracy",  metrics.get("test_accuracy"),  ACCURACY_THRESHOLD),
        ("test_precision", metrics.get("test_precision"), PRECISION_THRESHOLD),
        ("test_recall",    metrics.get("test_recall"),    RECALL_THRESHOLD),
        ("test_f1",        metrics.get("test_f1"),        F1_THRESHOLD),
    ]

    for name, value, threshold in checks:
        if value is None:
            print(f"  [SKIP] {name:<20} — not logged in this run")
            continue
        status = "PASS" if value >= threshold else "FAIL"
        indicator = "✓" if status == "PASS" else "✗"
        print(f"  [{status}] {indicator} {name:<20} = {value:.4f}  (threshold >= {threshold})")
        if status == "FAIL":
            violations.append(f"{name} = {value:.4f} is below threshold {threshold}")

    print(f"{'='*60}")

    if violations:
        print("\n  EVALUATION GATE: FAILED")
        print("\n  Violations:")
        for v in violations:
            print(f"    - {v}")
        print()
        sys.exit(1)
    else:
        print("\n  EVALUATION GATE: PASSED")
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate an MLFlow run against quality thresholds.')
    parser.add_argument('--run_id', type=str, required=True,
                        help='MLFlow run ID to evaluate.')
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLFlow tracking URI (overrides config.yml and env var).')
    args = parser.parse_args()

    config = load_config()
    tracking_uri = (
        args.tracking_uri
        or os.environ.get('MLFLOW_TRACKING_URI')
        or config.get('mlflow', {}).get('tracking_uri', '')
    )

    evaluate(args.run_id, tracking_uri)


if __name__ == "__main__":
    main()
