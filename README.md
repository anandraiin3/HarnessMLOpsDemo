# Harness Bank — Credit Card Approval MLOps Demo

End-to-end MLOps demonstration on Harness: model training with MLFlow, automated quality gates, Docker image builds, Kubernetes canary deployments on AWS EKS, and Dynatrace continuous verification.

---

## What the Application Does

**Harness Bank** is a fictional bank portal where customers apply for a credit card. An ML model (RandomForest) makes an instant approval or rejection decision based on the applicant's financial profile.

### Form Fields

| Field | Sent to ML model? | Notes |
|---|---|---|
| Full Name | No | UX only |
| Email Address | No | UX only |
| Annual Income ($) | **Yes** | Minimum $60,000 enforced client-side |
| Number of Children | **Yes** | Dropdown 0–6 |
| Own a Car? | **Yes** | Checkbox |
| Own a House? | **Yes** | Checkbox |
| Agree to T&Cs | No | Required to submit |

Only the 4 financial fields are sent to `/predict` on the Flask API.

### End-User Experience

1. User fills the form and clicks **Submit Application**
2. A dialog appears immediately with the result:

| Model output | Dialog | Message |
|---|---|---|
| `1` — Approved | Green success | "Congratulations, your application has been approved!" |
| `0` — Rejected | Yellow warning | "Unfortunately, your application was not approved this time." |

### Demo Scenarios (known inputs)

Use these to make the demo predictable and impactful:

| Scenario | Income | Children | Car | House | Expected |
|---|---|---|---|---|---|
| High earner with assets | $150,000 | 0 | Yes | Yes | **Approved** |
| Low earner with dependants | $65,000 | 4 | No | No | **Rejected** |
| Middle income, owns house | $90,000 | 1 | No | Yes | **Approved** |

These same inputs are used by the integration tests in Stage 5 of the Harness pipeline.

---

## Architecture

```
Next.js Frontend  →  Flask API  →  RandomForest model (loaded from MLFlow registry)
      ↑                  ↑
  (EKS pod)          (EKS pod)
```

### Components

| Component | Path | Description |
|---|---|---|
| Next.js frontend | `web_application/app/harness_bank_app/` | Credit card application form (MUI, TypeScript) |
| Flask prediction API | `web_application/api/app.py` | `/health` + `/predict` endpoints; gunicorn in production |
| ML training script | `harness/execute_mlflow_pipeline.py` | Trains RandomForest, logs to MLFlow, registers model |
| Model evaluation | `harness/evaluate_model.py` | Quality gate: asserts accuracy > 80%, F1/precision/recall > 75% |
| Kubernetes manifests | `web_application/k8s/` | EKS deployments, services, AWS ALB ingress |
| Harness pipeline | `pipelines/harness-mlops-pipeline.yaml` | 7-stage MLOps pipeline (see below) |

---

## Harness Pipeline — 7 Stages

```
Stage 1  Train Model          CI   Python container — runs execute_mlflow_pipeline.py
                                   Outputs: MLFLOW_RUN_ID
Stage 2  Evaluate Model       CI   Python container — runs evaluate_model.py
                                   Fails pipeline if accuracy < 80% or F1 < 75%
Stage 3  Build and Push       CI   BuildAndPushDockerRegistry — Flask API + Next.js images
                                   Tagged with pipeline.sequenceId
Stage 4  Deploy Staging       CD   K8s rolling deploy to mlops-staging namespace (EKS)
Stage 5  Integration Tests    CI   curl /health + smoke-test /predict with known inputs
Stage 6  Approval Gate        —    Manual approval (24h timeout) before production
Stage 7  Deploy Production    CD   K8s canary (25%) → Dynatrace CV verify → full rollout
```

### Pipeline Variables

| Variable | Default value | Description |
|---|---|---|
| `imageTag` | `<+pipeline.sequenceId>` | Docker image tag |
| `mlflowTrackingUri` | `https://mlflow.sandbox.harness-demo.site` | MLFlow server |
| `modelName` | `credit-card-approval` | Registered model name |
| `apiImageName` | `credit-card-api` | Flask API image |
| `appImageName` | `credit-card-app` | Next.js frontend image |

### Runtime Inputs (`<+input>`)

You will be prompted for these when running the pipeline:

- Kubernetes connector (EKS cluster)
- Docker registry connector + repository prefix
- Staging / production service refs and infrastructure IDs
- Delegate selectors

---

## AWS EKS Kubernetes Manifests

Located in `web_application/k8s/`:

| File | Purpose |
|---|---|
| `namespace.yaml` | `mlops`, `mlops-staging`, `mlops-production` namespaces |
| `api-deployment.yaml` | Flask API deployment + ClusterIP service |
| `app-deployment.yaml` | Next.js frontend deployment + ClusterIP service |
| `ingress.yaml` | AWS ALB ingress — routes `/health`, `/predict` → API, `/` → frontend |

The ALB ingress requires the [AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/) installed on the cluster. Update the `alb.ingress.kubernetes.io/certificate-arn` annotation with your ACM certificate ARN before deploying.

### Model Loading Priority (Flask API)

Set `MODEL_SOURCE` env var to control which source is used:

| `MODEL_SOURCE` | Behaviour |
|---|---|
| `auto` (default) | MLFlow registry → local file → S3 |
| `mlflow` | MLFlow registry only |
| `local` | Local file path (`MODEL_LOCAL_PATH`) |
| `s3` | S3 bucket (`S3_BUCKET` + `MODEL_PKL`) |

Required env vars for MLFlow: `MLFLOW_TRACKING_URI`, `MODEL_NAME`, optionally `MODEL_VERSION`.

---

## Local Development

### Run training
```bash
pip install -r requirements.txt
python harness/execute_mlflow_pipeline.py --n_estimators 100
# Run ID written to outputs/run_id.txt
```

### Run evaluation gate
```bash
python harness/evaluate_model.py --run_id <run_id> --tracking_uri https://mlflow.sandbox.harness-demo.site
```

### Run Flask API locally
```bash
cd web_application/api
MODEL_LOCAL_PATH=../../outputs/selected_model.joblib python app.py
curl localhost:5000/health
curl -X POST localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"income":150000,"children":0,"ownCar":true,"ownHouse":true}'
```

### Run Next.js frontend locally
```bash
cd web_application/app/harness_bank_app
NEXT_PUBLIC_API_URL=http://localhost:5000 npm run dev
# Open http://localhost:3000
```

### Run notebook tests
```bash
pytest --nbval-lax credit_card_approval.ipynb --junitxml=report.xml
```
