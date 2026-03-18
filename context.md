# Project Context — Harness Bank MLOps Demo

Last updated: 2026-03-18 (restructured to best-practice folder layout)

> For full design rationale see **[DESIGN.md](./DESIGN.md)**

## What This Is
End-to-end MLOps demo on the Harness platform for a fictional bank ("Harness Bank").
Customers apply for a credit card; a RandomForest ML model instantly approves or rejects
based on 4 financial features. Designed to showcase Harness CI/CD + AWS native MLOps.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | RandomForest (scikit-learn), trained on `synthetic_credit_card_approval.csv` |
| Experiment tracking | AWS SageMaker Experiments |
| Model registry | AWS SageMaker Model Registry |
| Artifact storage | AWS S3 |
| Prediction API | Flask + gunicorn (`web_application/api/app.py`) |
| Frontend | Next.js + TypeScript + MUI (`web_application/app/harness_bank_app/`) |
| Container runtime | AWS EKS (Kubernetes) |
| Ingress | AWS ALB (requires AWS Load Balancer Controller + ACM cert ARN) |
| CD verification | Dynatrace continuous verification (canary stage) |
| Pipeline | Harness (`pipelines/harness-mlops-pipeline.yaml`) |

---

## Model Features (in order expected by Flask API)
`[Num_Children, Income, Own_Car, Own_Housing]`

- `Own_Car` and `Own_Housing` are 0/1 integers
- Feature order is critical — must match training in `harness/train_model.py`
- `Group` column exists in the CSV for fairness metrics only — it is NOT a training feature

---

## Harness Pipeline — 8 Stages

| # | Stage | Type | Key action |
|---|---|---|---|
| 1 | Train Model | CI | `harness/train_model.py` → SageMaker Experiments + S3 + Model Registry (PendingManualApproval) |
| 2 | Evaluate Model | CI | `harness/evaluate_model.py` → reads metrics.json from S3, asserts thresholds |
| 3 | Build and Push Images | CI | BuildAndPushDockerRegistry for Flask API + Next.js |
| 4 | Deploy Staging | CD | K8s rolling deploy to `mlops-staging` namespace |
| 5 | Integration Tests | CI | curl `/health` + Python smoke tests on `/predict` |
| 6 | Approve Production Deployment | Approval | Manual gate (24h timeout) — shows run name, ARN, S3 metrics URI |
| 7 | Approve Model in Registry | CI | `harness/promote_model.py` → sets SageMaker Model Registry status to Approved |
| 8 | Deploy Production | CD | Canary 25% → Dynatrace CV (10m, MEDIUM sensitivity) → full rolling rollout |

### Pipeline Variables
- `imageTag` — `<+pipeline.sequenceId>` (auto)
- `awsRegion` — `us-east-1`
- `s3BucketName` — `<+input>` (required at runtime)
- `modelPackageGroup` — `credit-card-approval`
- `apiImageName` — `credit-card-api`
- `appImageName` — `credit-card-app`

### Runtime Inputs (prompted when running)
- K8s connector (EKS)
- Docker registry connector + repo prefix
- Staging/production service refs and infrastructure IDs
- Delegate selectors

---

## Folder Structure

```
ml-ops/
├── configs/                        # All config files (no secrets)
│   └── config.yml                  # Merged config (data paths, MLFlow, SageMaker, AWS)
├── data/
│   └── raw/                        # Immutable source data
│       └── synthetic_credit_card_approval.csv
├── deployment/
│   ├── api/                        # Flask prediction API
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── app/harness_bank_app/       # Next.js frontend
│   ├── k8s/                        # Kubernetes manifests (EKS)
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── app-deployment.yaml
│   │   ├── ingress.yaml
│   │   └── mlflow/                 # Optional self-hosted MLFlow
│   └── mlflow/Dockerfile           # MLFlow server image
├── notebooks/
│   └── credit_card_approval.ipynb  # Exploratory analysis only
├── outputs/                        # Gitignored generated artifacts
├── pipelines/
│   └── harness-mlops-pipeline.yaml
├── scripts/                        # Pipeline entrypoints (called by Harness stages)
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── promote_model.py
│   └── execute_mlflow_pipeline.py  # Legacy MLFlow training (local dev)
├── requirements.txt
└── pyproject.toml
```

---

## Key Scripts

### `scripts/train_model.py`
- Trains RandomForest, logs to SageMaker Experiments
- Uploads `model.joblib` and `metrics.json` to S3 at `s3://<bucket>/models/<group>/<run_name>/`
- Registers model in SageMaker Model Registry as `PendingManualApproval`
- Writes to `outputs/`: `run_name.txt`, `model_package_arn.txt`, `s3_metrics_uri.txt`
- Env vars required: `S3_BUCKET`, `AWS_DEFAULT_REGION`

### `scripts/evaluate_model.py`
- Downloads `metrics.json` from S3, asserts thresholds, exits non-zero on failure
- Thresholds: accuracy ≥ 80%, precision/recall/F1 ≥ 75%, fairness gap < 20%
- Args: `--s3_metrics_uri`, `--region`

### `scripts/promote_model.py`
- Sets SageMaker Model Registry package status to `Approved`
- Args: `--model_package_arn`, `--region`

### `scripts/execute_mlflow_pipeline.py`
- **Legacy MLFlow-based training script** (still present, was the original Stage 1)
- Uses MLFlow registry instead of SageMaker — the pipeline now uses `train_model.py`
- May still be used for local dev/testing with a self-hosted MLFlow server

---

## Flask API (`deployment/api/app.py`)

### Model Loading Priority
- `MODEL_SOURCE=auto` (default): SageMaker Model Registry → local file
- `MODEL_SOURCE=sagemaker`: SageMaker only (latest Approved package)
- `MODEL_SOURCE=local`: uses `MODEL_LOCAL_PATH`
- Key env vars: `SAGEMAKER_MODEL_PACKAGE_GROUP`, `AWS_DEFAULT_REGION`, `MODEL_LOCAL_PATH`

### Endpoints
- `GET /health` — returns `{"status": "ok", "model_loaded": bool}`
- `POST /predict` — body: `{"income": int, "children": int, "ownCar": bool, "ownHouse": bool}`
  - Returns: `{"prediction": 0|1, "result_str": "Approved|Rejected", "message": "..."}`

---

## Known-Good Demo Inputs

| Scenario | income | children | ownCar | ownHouse | Expected |
|---|---|---|---|---|---|
| High earner with assets | 150000 | 0 | true | true | Approved (1) |
| Low earner with dependants | 65000 | 4 | false | false | Rejected (0) |
| Middle income, owns house | 90000 | 1 | false | true | Approved (1) |

Used in Stage 5 integration tests (low-income test uses income=30000 in the pipeline YAML).

---

## Kubernetes Layout

```
deployment/k8s/
  namespace.yaml          # mlops, mlops-staging, mlops-production namespaces
  api-deployment.yaml     # Flask API deployment + ClusterIP service
  app-deployment.yaml     # Next.js frontend deployment + ClusterIP service
  ingress.yaml            # AWS ALB ingress — /health /predict → API, / → frontend
  mlflow/
    mlflow-server.yaml    # Self-hosted MLFlow server (optional, for local dev)
    postgres.yaml         # Postgres backend for MLFlow
```

---

## Config (`configs/config.yml`)
```yaml
data.load_file_path: data/raw/synthetic_credit_card_approval.csv
model.model_output_file_path: outputs
model.model_output_file: selected_model.joblib
mlflow.experiment_name: Credit Card Approval
mlflow.tracking_uri: https://mlflow.sandbox.harness-demo.site
sagemaker.experiment_name: credit-card-approval
aws.region: us-east-1
aws.s3_bucket: ''   # set via S3_BUCKET env var
```

---

## Local Dev

### Python environment (uv)
```bash
# First time setup
uv sync --dev          # creates .venv, installs all deps + dev deps

# Activate (optional — uv run works without activating)
source .venv/bin/activate

# Run a script
uv run python scripts/execute_mlflow_pipeline.py --n_estimators 100

# Run notebook
uv run jupyter lab

# Run tests
uv run pytest
```

### Three requirements files
| File | Used by | Contents |
|---|---|---|
| `pyproject.toml` | uv (local dev) | Core deps + dev group (jupyter, pytest, matplotlib) |
| `requirements.txt` | Harness CI pipeline (`pip install`) | Core deps only (no dev tools) |
| `deployment/api/requirements.txt` | Docker container | Flask, gunicorn, boto3, sagemaker |

### Train and test locally
```bash
# Train with MLFlow (no AWS needed)
uv run python scripts/execute_mlflow_pipeline.py --n_estimators 100

# Train with SageMaker (requires AWS credentials)
S3_BUCKET=my-bucket uv run python scripts/train_model.py

# Run Flask API
cd deployment/api
MODEL_SOURCE=local MODEL_LOCAL_PATH=../../outputs/model.joblib uv run python app.py

# Run Next.js frontend
cd deployment/app/harness_bank_app
NEXT_PUBLIC_API_URL=http://localhost:5000 npm run dev
```

---

## Restructure Applied (2026-03-18)
- `harness/` → `scripts/`
- `web_application/api/` → `deployment/api/`
- `web_application/app/harness_bank_app/` → `deployment/app/harness_bank_app/`
- `web_application/k8s/` → `deployment/k8s/`
- `mlflow/` → `deployment/mlflow/`
- `config.yml` + `harness/config.yml` → merged `configs/config.yml`
- `synthetic_credit_card_approval.csv` → `data/raw/`
- `credit_card_approval.ipynb` → `notebooks/`
- All script/pipeline path references updated accordingly

## Files Intentionally Removed (2026-03-18)
- `lambda_function.py` — old AWS Lambda inference approach (replaced by Flask/EKS)
- `Dockerfile_Inference_Lambda`, `Dockerfile_Training_Testing` — Lambda Dockerfiles
- `main.py` — empty placeholder
- `model_testing/` — scratch directory using Iris dataset / old DecisionTree
- `web_application/app/deployment.yaml` — stale single-file manifest (superseded by `k8s/`)
- `web_application/model_metrics.html` — generated output artifact
- `MLproject`, `harness/conda.yaml` — MLFlow project runner files (pipeline calls scripts directly)
