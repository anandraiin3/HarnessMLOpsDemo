"""
promote_model.py — Approve a model package in Amazon SageMaker Model Registry.

Sets the ModelApprovalStatus to 'Approved', which:
  1. Signals to the Flask API to load this model version on next startup
  2. Provides an immutable audit trail in the SageMaker console

AWS Services used:
  - SageMaker Model Registry : update approval status

Usage:
    python harness/promote_model.py --model_package_arn arn:aws:sagemaker:...
"""

import argparse
import boto3
import os
import sys


def promote(model_package_arn, region):
    sm = boto3.client('sagemaker', region_name=region)

    print(f"\n{'='*60}")
    print(f"  SageMaker Model Registry — Promoting to Approved")
    print(f"{'='*60}")
    print(f"  ARN: {model_package_arn}")

    # Show current status
    info = sm.describe_model_package(ModelPackageName=model_package_arn)
    current = info.get('ModelApprovalStatus', 'Unknown')
    print(f"  Current status: {current} → Approved")

    if current == 'Approved':
        print("  [INFO] Already Approved — no change needed")
        print(f"{'='*60}\n")
        return

    sm.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus='Approved',
    )

    print(f"\n  [DONE] Model package approved.")
    print(f"  Flask API pods will load this model on next startup.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Approve a model package in SageMaker Model Registry.'
    )
    parser.add_argument('--model_package_arn', required=True,
                        help='SageMaker Model Package ARN to approve')
    parser.add_argument('--region', type=str, default=None,
                        help='AWS region (default: AWS_DEFAULT_REGION env var)')
    args = parser.parse_args()

    region = args.region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    promote(args.model_package_arn, region)


if __name__ == "__main__":
    main()
