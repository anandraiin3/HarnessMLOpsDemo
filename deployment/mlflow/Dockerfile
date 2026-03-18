##############################################################################
# MLFlow Tracking Server — self-hosted on EKS
# Backend store : PostgreSQL (via psycopg2-binary)
# Artifact store: S3       (via boto3)
#
# Runtime env vars required:
#   MLFLOW_DB_USER, MLFLOW_DB_PASSWORD, MLFLOW_DB_HOST,
#   MLFLOW_DB_NAME, MLFLOW_S3_BUCKET
##############################################################################
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

EXPOSE 5000

CMD mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri postgresql://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_HOST}:5432/${MLFLOW_DB_NAME} \
      --default-artifact-root s3://${MLFLOW_S3_BUCKET}/mlflow-artifacts \
      --serve-artifacts
