"""
train_model.py — Train RandomForest, track with Amazon SageMaker Experiments,
                 save artifact to S3, register in SageMaker Model Registry.

AWS Services used:
  - SageMaker Experiments : experiment tracking (params, metrics, fairness)
  - S3                    : model artifact and metrics storage
  - SageMaker Model Registry : model versioning and approval workflow

IAM permissions required on the EKS pod (via IRSA or AWS credentials):
  s3:PutObject, s3:GetObject
  sagemaker:CreateExperiment, sagemaker:CreateTrial, sagemaker:CreateTrialComponent
  sagemaker:CreateModelPackageGroup, sagemaker:CreateModelPackage

Usage:
    python harness/train_model.py --n_estimators 100 --model_name credit-card-approval
"""

import argparse
import json
import os
import yaml
import boto3
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import sagemaker
    from sagemaker.experiments.run import Run as SageMakerRun
    SAGEMAKER_SDK = True
except ImportError:
    SAGEMAKER_SDK = False


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


# ── AWS helpers ────────────────────────────────────────────────────────────────

def log_to_sagemaker_experiments(experiment_name, run_name, params, metrics, region):
    """Log params and metrics to SageMaker Experiments (visible in AWS console)."""
    if not SAGEMAKER_SDK:
        print("[SKIP] sagemaker SDK not available — skipping Experiments logging")
        return
    try:
        session = sagemaker.Session(boto3.Session(region_name=region))
        with SageMakerRun(
            experiment_name=experiment_name,
            run_name=run_name,
            sagemaker_session=session,
        ) as sm_run:
            for k, v in params.items():
                sm_run.log_parameter(k, str(v))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    sm_run.log_metric(k, v)
        print(f"==> Logged to SageMaker Experiments: {experiment_name}/{run_name}")
    except Exception as e:
        print(f"[WARN] SageMaker Experiments logging failed (non-fatal): {e}")


def upload_to_s3(local_path, bucket, s3_key, region):
    s3 = boto3.client('s3', region_name=region)
    s3.upload_file(local_path, bucket, s3_key)
    uri = f"s3://{bucket}/{s3_key}"
    print(f"==> Uploaded to {uri}")
    return uri


def register_in_model_registry(run_name, s3_model_uri, model_package_group,
                                api_image_uri, region):
    """
    Register the model in SageMaker Model Registry as PendingManualApproval.
    The model will be promoted to Approved after the Harness approval gate passes.
    """
    sm = boto3.client('sagemaker', region_name=region)

    # Create model package group (idempotent)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group,
            ModelPackageGroupDescription="Credit card approval model versions",
        )
        print(f"==> Created model package group: {model_package_group}")
    except Exception:
        pass  # Group already exists

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus='PendingManualApproval',
        ModelPackageDescription=f"Training run: {run_name}",
        CustomerMetadataProperties={
            'run_name': run_name,
            's3_model_uri': s3_model_uri,
        },
        InferenceSpecification={
            'Containers': [
                {
                    'Image': api_image_uri,
                    'ModelDataUrl': s3_model_uri,
                }
            ],
            'SupportedContentTypes': ['application/json'],
            'SupportedResponseMIMETypes': ['application/json'],
        },
    )

    arn = response['ModelPackageArn']
    print(f"==> Registered in SageMaker Model Registry: {arn}")
    return arn


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train RandomForest with AWS tracking.')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='credit-card-approval')
    args = parser.parse_args()

    with open("configs/config.yml") as f:
        config = yaml.safe_load(f)

    bucket              = os.environ.get('S3_BUCKET',
                            config.get('aws', {}).get('s3_bucket', ''))
    region              = os.environ.get('AWS_DEFAULT_REGION',
                            config.get('aws', {}).get('region', 'us-east-1'))
    model_package_group = os.environ.get('SAGEMAKER_MODEL_PACKAGE_GROUP', args.model_name)
    api_image_uri       = os.environ.get('API_IMAGE_URI', 'placeholder/credit-card-api:latest')
    experiment_name     = config.get('sagemaker', {}).get(
                            'experiment_name', 'credit-card-approval')

    if not bucket:
        raise ValueError("S3_BUCKET env var or aws.s3_bucket in config.yml is required")

    run_name = f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
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
        "n_estimators":      args.n_estimators,
        "n_features":        X.shape[1],
        "feature_names":     list(X.columns),
        "feature_importances": importances,
    }

    # ── Log to SageMaker Experiments ───────────────────────────────────────────
    log_to_sagemaker_experiments(experiment_name, run_name, params, all_metrics, region)

    # ── Save and upload artifacts ──────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)

    metrics_path = "outputs/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    model_path = "outputs/model.joblib"
    joblib.dump(model, model_path)

    s3_prefix      = f"models/{model_package_group}/{run_name}"
    s3_metrics_uri = upload_to_s3(metrics_path, bucket, f"{s3_prefix}/metrics.json", region)
    s3_model_uri   = upload_to_s3(model_path,   bucket, f"{s3_prefix}/model.joblib", region)

    # ── Register in SageMaker Model Registry ──────────────────────────────────
    model_package_arn = register_in_model_registry(
        run_name, s3_model_uri, model_package_group, api_image_uri, region
    )

    # ── Write outputs for downstream Harness pipeline steps ───────────────────
    with open("outputs/run_name.txt", "w") as f:
        f.write(run_name)
    with open("outputs/model_package_arn.txt", "w") as f:
        f.write(model_package_arn)
    with open("outputs/s3_metrics_uri.txt", "w") as f:
        f.write(s3_metrics_uri)

    print(f"\n==> Run name         : {run_name}")
    print(f"==> Train accuracy   : {train_metrics['accuracy']:.4f}")
    print(f"==> Test  accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"==> Fairness gap     : {fairness['fairness_gap']:.4f}")
    print(f"==> Model package ARN: {model_package_arn}")
    print(f"==> Metrics at       : {s3_metrics_uri}")


if __name__ == "__main__":
    main()
