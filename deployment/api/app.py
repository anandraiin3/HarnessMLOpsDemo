import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Model loading ──────────────────────────────────────────────────────────────
# Priority: SageMaker Model Registry (latest Approved) → local file
# Override by setting MODEL_SOURCE = "sagemaker" | "local"

model = None


def load_model_from_sagemaker_registry(model_package_group, region):
    """
    Fetch the latest Approved model package from SageMaker Model Registry,
    then download the model artifact from its S3 URI.
    """
    import boto3, joblib
    sm = boto3.client('sagemaker', region_name=region)

    response = sm.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1,
    )

    packages = response.get('ModelPackageSummaryList', [])
    if not packages:
        raise RuntimeError(
            f"No Approved model in SageMaker Model Registry group: {model_package_group}"
        )

    arn = packages[0]['ModelPackageArn']
    info = sm.describe_model_package(ModelPackageName=arn)

    # Prefer the CustomerMetadataProperties URI (direct joblib), fall back to container URI
    s3_model_uri = (
        info.get('CustomerMetadataProperties', {}).get('s3_model_uri')
        or info['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    )
    print(f"[INFO] Loading Approved model from: {s3_model_uri}")

    without_prefix = s3_model_uri.replace("s3://", "")
    bucket, key = without_prefix.split("/", 1)

    s3 = boto3.client('s3', region_name=region)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp_path = tmp.name

    return joblib.load(tmp_path)


def load_model_from_local(path):
    import joblib
    print(f"[INFO] Loading model from local file: {path}")
    return joblib.load(path)


def load_model():
    source              = os.environ.get("MODEL_SOURCE", "auto").lower()
    model_package_group = os.environ.get("SAGEMAKER_MODEL_PACKAGE_GROUP", "credit-card-approval")
    region              = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    local_path          = os.environ.get("MODEL_LOCAL_PATH", "")

    if source == "sagemaker" or (source == "auto" and model_package_group):
        return load_model_from_sagemaker_registry(model_package_group, region)
    if source == "local" or (source == "auto" and local_path):
        return load_model_from_local(local_path)

    raise RuntimeError(
        "No model source configured. Set SAGEMAKER_MODEL_PACKAGE_GROUP or MODEL_LOCAL_PATH."
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })


@app.route("/predict", methods=["OPTIONS", "POST"])
def predict():
    """Accept applicant data, return approval prediction."""
    if request.method == "OPTIONS":
        response = app.response_class()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data         = request.json
    income       = data.get("income", 0)
    num_children = data.get("children", 0)
    own_car      = 1 if data.get("ownCar", False) else 0
    own_house    = 1 if data.get("ownHouse", False) else 0

    print(f"[DEBUG] income={income}, children={num_children}, car={own_car}, house={own_house}")

    # Feature order matches training: Num_Children, Income, Own_Car, Own_Housing
    features = np.array([
        int(num_children),
        int(income),
        int(own_car),
        int(own_house),
    ]).reshape(1, -1)

    prediction = int(model.predict(features)[0])
    result_str = "Approved" if prediction == 1 else "Rejected"

    print(f"[DEBUG] prediction={prediction} ({result_str})")

    return jsonify({
        "prediction": prediction,
        "result_str": result_str,
        "message":    f"Model Output: {prediction} ({result_str})",
    })


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        model = load_model()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load model at startup: {e}")
        print("[WARNING] Server will start — /predict returns 503 until model is available.")
    app.run(host="0.0.0.0", port=port)
