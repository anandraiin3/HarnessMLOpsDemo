"""
log_experiment.py — Log training run parameters and metrics to Amazon SageMaker Experiments.

Reads artifacts written by train_model.py from the shared pipeline workspace
(outputs/run_name.txt, outputs/metrics.json, outputs/params.json) and logs
them to the SageMaker Experiments console for experiment tracking and comparison.

AWS Services used:
  - SageMaker Experiments : parameter and metric tracking

IAM permissions required:
  sagemaker:CreateExperiment, sagemaker:CreateTrial, sagemaker:CreateTrialComponent

Usage:
    python scripts/log_experiment.py \
        --experiment_name credit-card-approval \
        --region us-east-1

All arguments are required — values are passed from Harness pipeline variables
so that external system configuration is owned and visible in the pipeline,
not buried as defaults inside the script.
"""

import argparse
import json
import os
import sys

import boto3

try:
    import sagemaker
    from sagemaker.experiments.run import Run as SageMakerRun
    SAGEMAKER_SDK = True
except ImportError:
    SAGEMAKER_SDK = False


def main():
    parser = argparse.ArgumentParser(description='Log run to SageMaker Experiments.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='SageMaker Experiments name (passed from Harness pipeline variable)')
    parser.add_argument('--region', type=str, required=True,
                        help='AWS region (passed from Harness pipeline variable)')
    args = parser.parse_args()

    region = args.region

    # Read artifacts from shared workspace (written by train_model.py)
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

    if not SAGEMAKER_SDK:
        print("\n  [SKIP] sagemaker SDK not available — install with: pip install sagemaker")
        sys.exit(1)

    try:
        session = sagemaker.Session(boto3.Session(region_name=region))
        with SageMakerRun(
            experiment_name=args.experiment_name,
            run_name=run_name,
            sagemaker_session=session,
        ) as sm_run:
            for k, v in params.items():
                sm_run.log_parameter(k, str(v))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    sm_run.log_metric(k, v)

        print(f"\n  [DONE] Run logged to SageMaker Experiments.")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  [ERROR] SageMaker Experiments logging failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
