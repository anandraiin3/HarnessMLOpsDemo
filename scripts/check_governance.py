"""
check_governance.py — Enforce OPA model governance policy via Harness pipeline.

Downloads model metrics from S3 and evaluates them against the Rego policy
in policies/model_governance.rego using the OPA CLI. The pipeline halts if
any deny rule fires, keeping the model in PendingManualApproval.

This demonstrates Harness Governance: policy-as-code enforced automatically
at every pipeline run, with a full audit trail in Harness execution history.

AWS Services used:
  - S3 : download metrics.json for policy evaluation

Prerequisites (installed in pipeline step):
  - pip install boto3
  - OPA binary at /usr/local/bin/opa

Usage:
    python scripts/check_governance.py \
        --s3_metrics_uri s3://bucket/models/credit-card-approval/run-xxx/metrics.json \
        --policy_file policies/model_governance.rego \
        --region us-east-1
"""

import argparse
import json
import os
import subprocess
import sys

import boto3


def download_metrics(s3_uri, region):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    s3 = boto3.client('s3', region_name=region)
    body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return json.loads(body)


def run_opa(input_file, policy_file, query):
    result = subprocess.run(
        [
            "opa", "eval",
            "--format", "json",
            "--input", input_file,
            "--data", policy_file,
            query,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] OPA evaluation failed:\n{result.stderr}")
        sys.exit(1)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description='Enforce OPA model governance policy.'
    )
    parser.add_argument('--s3_metrics_uri', required=True,
                        help='S3 URI of metrics.json from the training run')
    parser.add_argument('--policy_file', default='policies/model_governance.rego',
                        help='Path to the Rego policy file')
    parser.add_argument('--region', type=str, default=None)
    args = parser.parse_args()

    region = args.region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

    # ── Download metrics from S3 ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Harness Governance — OPA Policy Evaluation")
    print(f"{'='*60}")
    print(f"  Policy : {args.policy_file}")
    print(f"  Metrics: {args.s3_metrics_uri}")
    print()

    metrics = download_metrics(args.s3_metrics_uri, region)

    print(f"  Model metrics:")
    print(f"    test_accuracy  : {metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"    test_precision : {metrics.get('test_precision', 'N/A'):.4f}")
    print(f"    test_recall    : {metrics.get('test_recall', 'N/A'):.4f}")
    print(f"    test_f1        : {metrics.get('test_f1', 'N/A'):.4f}")
    print(f"    fairness_gap   : {metrics.get('fairness_gap', 'N/A'):.4f}")

    # ── Write input file for OPA ───────────────────────────────────────────────
    input_file = "/tmp/model_metrics.json"
    with open(input_file, "w") as f:
        json.dump(metrics, f)

    # ── Evaluate deny rules ────────────────────────────────────────────────────
    output = run_opa(input_file, args.policy_file, "data.mlops.model_governance.deny")

    # OPA json output: {"result": [{"expressions": [{"value": [...]}]}]}
    violations = output["result"][0]["expressions"][0]["value"]

    print(f"\n{'='*60}")
    if violations:
        print(f"  GOVERNANCE GATE: FAILED")
        print(f"\n  Policy violations ({len(violations)}):")
        for v in violations:
            print(f"    x {v}")
        print(f"\n  Model remains PendingManualApproval in SageMaker Model Registry.")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  GOVERNANCE GATE: PASSED")
        print(f"\n  All policy checks satisfied — model cleared for quality evaluation.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
