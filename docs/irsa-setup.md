# AWS IRSA Setup for Harness MLOps Pipeline

IRSA (IAM Roles for Service Accounts) lets EKS pods assume an IAM role via a projected service account token. This replaces static `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` credentials — boto3 picks up temporary STS credentials automatically with no code changes.

---

## What gets access to what

Every pipeline stage that calls AWS runs as a pod using the `helm-delegate-mlops` service account in the `harness-delegate-ng` namespace. IRSA maps that service account to a single IAM role.

| Stage | Script | AWS Service | Operations |
|---|---|---|---|
| 1 – Step 2 | `log_experiment.py` | SageMaker Experiments | CreateExperiment, CreateTrial, CreateTrialComponent, AssociateTrialComponent, UpdateTrialComponent, BatchPutMetrics |
| 1 – Step 3/4 | Harness S3Upload | S3 | PutObject, GetBucketLocation |
| 1 – Step 5 | `register_model.py` | SageMaker Model Registry | CreateModelPackageGroup, CreateModelPackage |
| 2 – Step 1 | inline Python | S3 | GetObject |
| 3 | `evaluate_model.py` | S3 | GetObject |
| 8 | `promote_model.py` | SageMaker Model Registry | DescribeModelPackage, UpdateModelPackage |

---

## Prerequisites

- EKS cluster running in `ap-southeast-2`
- `kubectl` configured against the cluster
- AWS CLI configured with permissions to create IAM roles and policies
- OIDC provider already registered: `arn:aws:iam::282764120287:oidc-provider/oidc.eks.ap-southeast-2.amazonaws.com/id/0A760171068D9925E6C132A94407699D`

---

## Step 1 — Create the SageMaker IAM policy

```bash
cat > harness-mlops-sagemaker-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerExperiments",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateExperiment",
        "sagemaker:DescribeExperiment",
        "sagemaker:UpdateExperiment",
        "sagemaker:CreateTrial",
        "sagemaker:DescribeTrial",
        "sagemaker:UpdateTrial",
        "sagemaker:CreateTrialComponent",
        "sagemaker:DescribeTrialComponent",
        "sagemaker:UpdateTrialComponent",
        "sagemaker:AssociateTrialComponent",
        "sagemaker:DisassociateTrialComponent",
        "sagemaker:BatchPutMetrics"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SageMakerModelRegistry",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModelPackageGroup",
        "sagemaker:ListModelPackageGroups",
        "sagemaker:CreateModelPackage",
        "sagemaker:DescribeModelPackage",
        "sagemaker:UpdateModelPackage",
        "sagemaker:ListModelPackages"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name harness-mlops-sagemaker-policy \
  --policy-document file://harness-mlops-sagemaker-policy.json \
  --region ap-southeast-2
```

---

## Step 2 — Create the S3 IAM policy

Replace `<YOUR_S3_BUCKET>` with your actual bucket name.

```bash
cat > harness-mlops-s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ModelArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::<YOUR_S3_BUCKET>/models/*"
    },
    {
      "Sid": "S3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::<YOUR_S3_BUCKET>"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name harness-mlops-s3-policy \
  --policy-document file://harness-mlops-s3-policy.json \
  --region ap-southeast-2
```

---

## Step 3 — Create the IAM role with a trust policy

The trust policy binds the role to the `helm-delegate-mlops` service account in the `harness-delegate-ng` namespace. Only pods running as that service account can assume this role.

```bash
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::282764120287:oidc-provider/oidc.eks.ap-southeast-2.amazonaws.com/id/0A760171068D9925E6C132A94407699D"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.ap-southeast-2.amazonaws.com/id/0A760171068D9925E6C132A94407699D:sub": "system:serviceaccount:harness-delegate-ng:helm-delegate-mlops",
          "oidc.eks.ap-southeast-2.amazonaws.com/id/0A760171068D9925E6C132A94407699D:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name harness-mlops-pipeline-role \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name harness-mlops-pipeline-role \
  --policy-arn arn:aws:iam::282764120287:policy/harness-mlops-sagemaker-policy

aws iam attach-role-policy \
  --role-name harness-mlops-pipeline-role \
  --policy-arn arn:aws:iam::282764120287:policy/harness-mlops-s3-policy
```

---

## Step 4 — Annotate the Kubernetes service account

```bash
kubectl annotate serviceaccount helm-delegate-mlops \
  -n harness-delegate-ng \
  eks.amazonaws.com/role-arn=arn:aws:iam::282764120287:role/harness-mlops-pipeline-role

# Verify
kubectl get serviceaccount helm-delegate-mlops \
  -n harness-delegate-ng \
  -o yaml
```

Expected annotation in the output:
```yaml
annotations:
  eks.amazonaws.com/role-arn: arn:aws:iam::282764120287:role/harness-mlops-pipeline-role
```

---

## Step 5 — Restart the delegate

The IRSA token is only mounted at pod start time.

```bash
kubectl rollout restart deployment/helm-delegate-mlops -n harness-delegate-ng
kubectl rollout status deployment/helm-delegate-mlops -n harness-delegate-ng
```

---

## Step 6 — Verify IRSA is active

```bash
kubectl exec -n harness-delegate-ng \
  $(kubectl get pod -n harness-delegate-ng \
    -l app.kubernetes.io/name=helm-delegate-mlops \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}') \
  -- env | grep AWS_
```

Expected output:
```
AWS_REGION=ap-southeast-2
AWS_DEFAULT_REGION=ap-southeast-2
AWS_ROLE_ARN=arn:aws:iam::282764120287:role/harness-mlops-pipeline-role
AWS_WEB_IDENTITY_TOKEN_FILE=/var/run/secrets/eks.amazonaws.com/serviceaccount/token
AWS_STS_REGIONAL_ENDPOINTS=regional
```

`AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` together are what boto3 uses to exchange the projected token for temporary STS credentials. No code changes are needed in any pipeline script.

---

## Step 7 — Update the Harness AWS connector (for S3Upload steps)

The native `S3Upload` steps (Stage 1, Steps 3 and 4) use a Harness AWS connector. To route those through IRSA instead of static keys:

1. **Harness → Account Settings → Connectors → your AWS connector**
2. Set **Credential Type** to `IAM Role`
3. Enable **Assume IAM Role on Delegate**
4. Clear any Access Key / Secret Key fields

The delegate pod already has `AWS_ROLE_ARN` set, so Harness will call `sts:AssumeRoleWithWebIdentity` transparently.

---

## Reference

| Resource | Value |
|---|---|
| AWS Account ID | `282764120287` |
| Region | `ap-southeast-2` |
| EKS OIDC Provider | `arn:aws:iam::282764120287:oidc-provider/oidc.eks.ap-southeast-2.amazonaws.com/id/0A760171068D9925E6C132A94407699D` |
| IAM Role | `arn:aws:iam::282764120287:role/harness-mlops-pipeline-role` |
| Kubernetes namespace | `harness-delegate-ng` |
| Service account | `helm-delegate-mlops` |
| Delegate deployment | `helm-delegate-mlops` |
