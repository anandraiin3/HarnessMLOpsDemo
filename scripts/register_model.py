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
        --region us-east-1
"""

import argparse
import os
import sys

import boto3


def main():
    parser = argparse.ArgumentParser(
        description='Register model in SageMaker Model Registry.'
    )
    parser.add_argument('--s3_model_uri', required=True,
                        help='S3 URI of the uploaded model artifact')
    parser.add_argument('--model_package_group', type=str,
                        default=os.environ.get('SAGEMAKER_MODEL_PACKAGE_GROUP',
                                               'credit-card-approval'))
    parser.add_argument('--api_image_uri', type=str,
                        default=os.environ.get('API_IMAGE_URI',
                                               'placeholder/credit-card-api:latest'))
    parser.add_argument('--region', type=str, default=None)
    args = parser.parse_args()

    region = args.region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

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
    except sm.exceptions.ClientError:
        pass  # Group already exists

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
                    'Image':        args.api_image_uri,
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
