"""
log_experiment.py — Log training run parameters and metrics to Amazon SageMaker Experiments.

Uses boto3 directly (no sagemaker SDK) to avoid the heavy dependency install
that caused context-canceled timeouts in the Harness CI step.

AWS Services used:
  - SageMaker Experiments : parameter and metric tracking

IAM permissions required:
  sagemaker:CreateExperiment, sagemaker:CreateTrial, sagemaker:CreateTrialComponent,
  sagemaker:AssociateTrialComponent, sagemaker:UpdateTrialComponent

Usage:
    python scripts/log_experiment.py \
        --experiment_name credit-card-approval \
        --region us-east-1
"""

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError


def create_if_not_exists(fn, **kwargs):
    try:
        fn(**kwargs)
    except ClientError as e:
        code = e.response['Error']['Code']
        msg = e.response['Error']['Message']
        if code not in ('ValidationException', 'ResourceInUse'):
            print(f"\n  [ERROR] AWS error: {code} — {msg}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Log run to SageMaker Experiments.')
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--region', required=True)
    args = parser.parse_args()

    with open("outputs/run_name.txt") as f:
        run_name = f.read().strip()
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    with open("outputs/params.json") as f:
        params = json.load(f)

    print(f"\n{'='*60}")
    print(f"  SageMaker Experiments — Logging Run")
    print(f"{'='*60}")
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Run        : {run_name}")
    print(f"  Metrics    : accuracy={metrics.get('test_accuracy', 0):.4f}, "
          f"fairness_gap={metrics.get('fairness_gap', 0):.4f}")

    sm = boto3.client('sagemaker', region_name=args.region)
    component_name = f"{run_name}-training"

    # Create experiment (idempotent)
    create_if_not_exists(
        sm.create_experiment,
        ExperimentName=args.experiment_name,
        Description="Credit card approval model training runs",
    )

    # Create trial (one per training run)
    create_if_not_exists(
        sm.create_trial,
        TrialName=run_name,
        ExperimentName=args.experiment_name,
    )

    # Create trial component (the individual training job)
    create_if_not_exists(
        sm.create_trial_component,
        TrialComponentName=component_name,
    )

    # Link component to trial
    create_if_not_exists(
        sm.associate_trial_component,
        TrialName=run_name,
        TrialComponentName=component_name,
    )

    # Log parameters and metric summaries
    sm.update_trial_component(
        TrialComponentName=component_name,
        Parameters={k: {'StringValue': str(v)} for k, v in params.items()},
        Metrics=[
            {
                'MetricName': k,
                'Min': float(v),
                'Max': float(v),
                'Last': float(v),
                'Avg': float(v),
                'Count': 1,
            }
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ],
    )

    print(f"\n  [DONE] Run logged to SageMaker Experiments.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
