"""
register_model.py — Register a trained model in Amazon SageMaker Model Registry.

Registers a new model package version as PendingManualApproval.
The S3 model URI is passed in as a CLI argument — the actual upload to S3
is handled by the Harness S3Upload step that runs before this script.

The model is promoted to Approved after the Harness approval gate passes
(handled by promote_model.py in a later pipeline stage).

AWS Services used:
  - SageMaker Model Registry : model versioning and approval workflow

IAM permissions required:
  sagemaker:CreateModelPackageGroup, sagemaker:CreateModelPackage

Outputs:
  outputs/model_package_arn.txt — ARN of the registered model package

Usage:
    python scripts/register_model.py \
        --s3_model_uri s3://bucket/models/credit-card-approval/run-xxx/model.joblib \
        --model_package_group credit-card-approval \
        --inference_image_uri 123456789.dkr.ecr.us-east-1.amazonaws.com/credit-card-api:latest \
        --region us-east-1

All arguments are required — values are passed from Harness pipeline variables
so that external system configuration is owned and visible in the pipeline,
not buried as defaults inside the script.
"""

import argparse
import os
import re
import sys

import boto3
from botocore.exceptions import ClientError


def main():
    parser = argparse.ArgumentParser(
        description='Register model in SageMaker Model Registry.'
    )
    parser.add_argument('--s3_model_uri', required=True,
                        help='S3 URI of the uploaded model artifact (from Harness S3Upload step)')
    parser.add_argument('--model_package_group', type=str, required=True,
                        help='SageMaker Model Registry package group name (Harness pipeline variable)')
    parser.add_argument('--inference_image_uri', type=str, required=True,
                        help='Container image URI for inference (Harness pipeline variable)')
    parser.add_argument('--region', type=str, required=True,
                        help='AWS region (Harness pipeline variable)')
    args = parser.parse_args()

    _ECR_RE = re.compile(r'^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/.+:.+$')
    if not _ECR_RE.match(args.inference_image_uri):
        print(
            f"\n  [ERROR] --inference_image_uri must be a private ECR image URI.\n"
            f"  Got    : {args.inference_image_uri}\n"
            f"  Expect : 123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/credit-card-api:tag\n"
            f"  Note   : public.ecr.aws and Docker Hub images are NOT accepted by SageMaker.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    region = args.region

    with open("outputs/run_name.txt") as f:
        run_name = f.read().strip()

    sm = boto3.client('sagemaker', region_name=region)

    print(f"\n{'='*60}")
    print(f"  SageMaker Model Registry — Registering Model Package")
    print(f"{'='*60}")
    print(f"  Group   : {args.model_package_group}")
    print(f"  Run     : {run_name}")
    print(f"  Model   : {args.s3_model_uri}")

    # Create model package group (idempotent)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=args.model_package_group,
            ModelPackageGroupDescription="Credit card approval model versions",
        )
        print(f"  Created model package group: {args.model_package_group}")
    except ClientError as e:
        if e.response['Error']['Code'] not in ('ResourceInUse', 'ConflictException'):
            raise

    response = sm.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelApprovalStatus='PendingManualApproval',
        ModelPackageDescription=f"Training run: {run_name}",
        CustomerMetadataProperties={
            'run_name':     run_name,
            's3_model_uri': args.s3_model_uri,
        },
        InferenceSpecification={
            'Containers': [
                {
                    'Image':        args.inference_image_uri,
                    'ModelDataUrl': args.s3_model_uri,
                }
            ],
            'SupportedContentTypes':      ['application/json'],
            'SupportedResponseMIMETypes': ['application/json'],
        },
    )

    arn = response['ModelPackageArn']

    with open("outputs/model_package_arn.txt", "w") as f:
        f.write(arn)

    print(f"\n  [DONE] Registered as PendingManualApproval.")
    print(f"  ARN: {arn}")
    print(f"  Model will be promoted to Approved after the Harness approval gate.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
