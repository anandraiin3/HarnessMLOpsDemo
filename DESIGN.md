# Design Document — Harness Bank MLOps Demo

**Version:** 1.0
**Date:** 2026-03-18
**Status:** Current

---

## 1. Problem Statement

### Business Problem
Banks manually review credit card applications, which is slow, inconsistent, and hard to scale. The goal is to automate instant approval/rejection decisions using an ML model trained on applicant financial data.

### Engineering Problem
Beyond building the model, the real challenge is **operationalising it safely**:

- How do you retrain and redeploy a model without downtime or regressions?
- How do you catch a bad model before it reaches production?
- How do you give a human a checkpoint before risky production changes?
- How do you roll back quickly if a deployment goes wrong?
- How do you track experiments, compare model versions, and audit decisions?

This project solves all of the above using a fully automated MLOps pipeline on Harness.

---

## 2. Solution Overview

An end-to-end MLOps pipeline that trains a RandomForest classifier, validates it against quality and fairness thresholds, builds and pushes Docker images, deploys to staging, runs integration tests, waits for human approval, then promotes the model and deploys to production using a canary rollout verified by Dynatrace.

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐
│  Train      │───▶│  Evaluate    │───▶│  Build Images │───▶│  Deploy Staging │
│  (Stage 1)  │    │  (Stage 2)   │    │  (Stage 3)    │    │  (Stage 4)      │
└─────────────┘    └──────────────┘    └───────────────┘    └─────────────────┘
                          │                                          │
                     Fail = pipeline                          ┌──────▼──────────┐
                     stops here                              │ Integration     │
                                                             │ Tests (Stage 5) │
                                                             └──────┬──────────┘
                                                                    │
                                                             ┌──────▼──────────┐
                                                             │ Approval Gate   │
                                                             │ (Stage 6)       │
                                                             └──────┬──────────┘
                                                                    │
                                                       ┌────────────▼────────────┐
                                                       │ Approve in Registry     │
                                                       │ (Stage 7)               │
                                                       └────────────┬────────────┘
                                                                    │
                                                       ┌────────────▼────────────┐
                                                       │ Deploy Production       │
                                                       │ Canary → CV → Rollout   │
                                                       │ (Stage 8)               │
                                                       └─────────────────────────┘
```

---

## 3. Architecture

### Components

```
Next.js Frontend  ──▶  Flask Prediction API  ──▶  RandomForest Model
     (EKS)                   (EKS)                  (loaded from SageMaker
                                                      Model Registry / S3)
                                 │
                          SageMaker Experiments
                          (metrics, fairness, params)
```

| Component | Technology | Location |
|---|---|---|
| ML model | RandomForest (scikit-learn) | Loaded at API startup from S3 via SageMaker Model Registry |
| Training script | Python | `scripts/train_model.py` |
| Evaluation gate | Python | `scripts/evaluate_model.py` |
| Model promotion | Python | `scripts/promote_model.py` |
| Prediction API | Flask + gunicorn | `deployment/api/` |
| Frontend | Next.js + TypeScript + MUI | `deployment/app/harness_bank_app/` |
| Experiment tracking | AWS SageMaker Experiments | AWS Console |
| Model registry | AWS SageMaker Model Registry | AWS Console |
| Artifact storage | AWS S3 | `s3://<bucket>/models/<group>/<run>/` |
| Container runtime | AWS EKS | `mlops`, `mlops-staging`, `mlops-production` namespaces |
| Ingress | AWS ALB | Routes `/health`, `/predict` → API; `/` → frontend |
| CD verification | Dynatrace | Canary stage, 10-minute window, MEDIUM sensitivity |
| Pipeline orchestration | Harness | `pipelines/harness-mlops-pipeline.yaml` |

---

## 4. Data

### Dataset
`data/raw/synthetic_credit_card_approval.csv` — a synthetic credit card applicant dataset.

### Input Features (used by model)
| Feature | Type | Notes |
|---|---|---|
| `Income` | int | Annual income in USD |
| `Num_Children` | int | Number of dependants |
| `Own_Car` | int (0/1) | Car ownership |
| `Own_Housing` | int (0/1) | Home ownership |

### Sensitive Attribute
- `Group` — demographic group label. Used **only** for fairness evaluation; deliberately excluded from training features (fairness-unaware model by design).

### Target
- `Target` — binary: `1` = Approved, `0` = Rejected

---

## 5. Model

### Algorithm: RandomForest
**Why RandomForest over other algorithms:**
- Interpretable feature importances — important for a financial decision model
- Robust to outliers and missing values
- No feature scaling required
- Strong out-of-the-box performance on tabular data with few features
- Avoids overfitting better than a single decision tree

### Training
- 80/20 train/test split, `random_state=42` for reproducibility
- Default: `n_estimators=100` (configurable via `--n_estimators` CLI arg)
- Feature order expected by the API: `[Num_Children, Income, Own_Car, Own_Housing]`

### Quality Thresholds (automated gate in Stage 2)
| Metric | Threshold | Rationale |
|---|---|---|
| Test accuracy | ≥ 80% | Minimum acceptable for a financial decision |
| Test precision | ≥ 75% | Limit false approvals (costly defaults) |
| Test recall | ≥ 75% | Limit false rejections (lost revenue) |
| Test F1 | ≥ 75% | Balanced measure |
| Fairness gap | < 20% | Max approval-rate disparity between demographic groups |

A pipeline run that fails these thresholds **stops at Stage 2** — no image is built, nothing is deployed.

---

## 6. Pipeline Design Decisions

### Decision 1: Quality Gate Before Build
The evaluate step (Stage 2) runs before the Docker build (Stage 3). This avoids wasting build time and registry storage on a model that will never be deployed.

### Decision 2: Model Registry with PendingManualApproval
The model is registered in SageMaker Model Registry as `PendingManualApproval` at training time. It is only set to `Approved` **after** the human approval gate passes (Stage 7). This means:
- The registry always reflects what has been human-reviewed
- The Flask API only loads `Approved` models
- There is an immutable audit trail of every model version reviewed

### Decision 3: Human Approval Gate
Stage 6 (Harness Approval) blocks production deployment until a human reviewer explicitly signs off. The approval message surfaces the SageMaker run name, model package ARN, S3 metrics URI, and image tag so reviewers have full context without leaving Harness.

This is intentional — fully automated deploys to production are appropriate for some systems but not for a financial credit decisioning model where a bad model has direct customer impact.

### Decision 4: Canary Deployment with Dynatrace CV
Production deploy uses canary (25% of traffic) before full rollout:
- If Dynatrace detects anomalies in the 10-minute verification window, the stage rolls back automatically
- Only if the canary passes does the full rolling deploy proceed
- This limits the blast radius of a bad deployment to 25% of users for at most 10 minutes

### Decision 5: Model Loading at API Startup
The Flask API loads the model once at startup from SageMaker Model Registry (latest `Approved` version). This means:
- No per-request model loading overhead
- Model version is pinned for the lifetime of the pod
- To deploy a new model version, pods are restarted (which happens naturally during K8s rolling deploy)

### Decision 6: Fairness as a First-Class Gate
Approval-rate parity across demographic groups is measured and enforced as an automated threshold (gap < 20%), not just logged. A model that performs well on accuracy but discriminates across groups will fail the evaluation gate. This is a deliberate product decision given the financial/regulatory context.

### Decision 7: SageMaker vs MLFlow
The pipeline uses AWS SageMaker Experiments and Model Registry (not MLFlow) for production tracking. The reasoning:
- Native integration with IAM, no additional server to operate
- SageMaker Model Registry has a built-in approval workflow that maps directly to the Harness gate
- Audit trail is in AWS Console alongside other AWS resources

MLFlow (`scripts/execute_mlflow_pipeline.py`) is retained for local development with a self-hosted MLFlow server (`deployment/k8s/mlflow/`), where SageMaker credentials may not be available.

---

## 7. API Design

### Endpoints

**`GET /health`**
- Used by K8s liveness/readiness probes and Stage 5 integration tests
- Returns `{"status": "ok", "model_loaded": bool}`
- Returns 200 even if model is not loaded (so the pod stays up); `model_loaded: false` signals degraded state

**`POST /predict`**
```json
// Request
{ "income": 150000, "children": 0, "ownCar": true, "ownHouse": true }

// Response
{ "prediction": 1, "result_str": "Approved", "message": "Model Output: 1 (Approved)" }
```
- Feature order passed to the model: `[Num_Children, Income, Own_Car, Own_Housing]`
- CORS enabled for all origins (frontend and API are separate deployments)

### Model Source Priority
Controlled by `MODEL_SOURCE` env var:
```
auto (default):  SageMaker Model Registry → local file
sagemaker:       SageMaker only
local:           MODEL_LOCAL_PATH env var
```

---

## 8. Kubernetes Layout

Three namespaces to isolate environments:
- `mlops` — CI pipeline workloads (training, evaluation, integration test pods)
- `mlops-staging` — staging deployment
- `mlops-production` — production deployment

AWS ALB ingress routes traffic:
- `/health`, `/predict` → Flask API (`credit-card-api` service)
- `/` → Next.js frontend (`credit-card-app` service)

---

## 9. Known Limitations & Future Considerations

| Limitation | Potential improvement |
|---|---|
| Model is retrained from scratch every pipeline run | Add incremental learning or transfer learning |
| No feature drift detection | Add Evidently AI or SageMaker Model Monitor |
| No online A/B testing beyond canary | Integrate with a feature flag system |
| Fairness measured on approval rate parity only | Add equalised odds, calibration across groups |
| Single model in registry | Support champion/challenger model comparison |
| No data versioning | Add DVC for `data/raw/` versioning |
| Frontend has no auth | Add Cognito or similar for a realistic demo |
