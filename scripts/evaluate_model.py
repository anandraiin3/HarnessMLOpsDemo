"""
evaluate_model.py — MLOps quality and fairness gate for the Harness pipeline.

Downloads the metrics.json produced by train_model.py from S3, asserts all
quality and fairness thresholds, and exits non-zero if any are violated.

AWS Services used:
  - S3 : read metrics.json uploaded during training

Usage:
    python harness/evaluate_model.py --s3_metrics_uri s3://bucket/models/.../metrics.json
"""

import argparse
import json
import os
import sys
import boto3


# ── Thresholds ────────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD     = 0.80   # Minimum test accuracy
PRECISION_THRESHOLD    = 0.75   # Minimum test precision
RECALL_THRESHOLD       = 0.75   # Minimum test recall
F1_THRESHOLD           = 0.75   # Minimum test F1
FAIRNESS_GAP_THRESHOLD = 0.20   # Maximum approval-rate gap between groups


def load_metrics_from_s3(s3_uri, region):
    """Download and parse metrics.json from S3."""
    without_prefix = s3_uri.replace("s3://", "")
    bucket, key = without_prefix.split("/", 1)
    s3 = boto3.client('s3', region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj['Body'].read())


def evaluate(metrics):
    print(f"\n{'='*60}")
    print(f"  MLOps Evaluation Gate (Amazon SageMaker)")
    print(f"{'='*60}")

    violations = []

    checks = [
        ("test_accuracy",  metrics.get("test_accuracy"),  ACCURACY_THRESHOLD,     ">="),
        ("test_precision", metrics.get("test_precision"), PRECISION_THRESHOLD,    ">="),
        ("test_recall",    metrics.get("test_recall"),    RECALL_THRESHOLD,       ">="),
        ("test_f1",        metrics.get("test_f1"),        F1_THRESHOLD,           ">="),
        ("fairness_gap",   metrics.get("fairness_gap"),   FAIRNESS_GAP_THRESHOLD, "<"),
    ]

    for name, value, threshold, op in checks:
        if value is None:
            print(f"  [SKIP] {name:<22} — not found in metrics")
            continue
        passed = (value >= threshold) if op == ">=" else (value < threshold)
        status = "PASS" if passed else "FAIL"
        indicator = "✓" if passed else "✗"
        print(f"  [{status}] {indicator} {name:<22} = {value:.4f}  "
              f"(threshold {op} {threshold})")
        if not passed:
            violations.append(f"{name} = {value:.4f} fails threshold {op} {threshold}")

    print(f"{'='*60}")

    if violations:
        print("\n  EVALUATION GATE: FAILED")
        print("\n  Violations:")
        for v in violations:
            print(f"    - {v}")
        print()
        sys.exit(1)
    else:
        print("\n  EVALUATION GATE: PASSED\n")


def main():
    parser = argparse.ArgumentParser(
        description='Assert quality and fairness thresholds from S3 metrics.'
    )
    parser.add_argument('--s3_metrics_uri', required=True,
                        help='S3 URI of metrics.json (s3://bucket/key/metrics.json)')
    parser.add_argument('--region', type=str, default=None,
                        help='AWS region (default: AWS_DEFAULT_REGION env var)')
    args = parser.parse_args()

    region = args.region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    metrics = load_metrics_from_s3(args.s3_metrics_uri, region)
    evaluate(metrics)


if __name__ == "__main__":
    main()
