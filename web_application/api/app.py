import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Model loading ──────────────────────────────────────────────────────────────
# Priority: MLFlow registry → local file → S3
# Override by setting MODEL_SOURCE = "mlflow" | "local" | "s3"

model = None


def load_model_from_mlflow(tracking_uri, model_name, model_version):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    stage_or_version = model_version or "Production"
    model_uri = f"models:/{model_name}/{stage_or_version}"
    print(f"[INFO] Loading model from MLFlow: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def load_model_from_local(path):
    import joblib
    print(f"[INFO] Loading model from local file: {path}")
    return joblib.load(path)


def load_model_from_s3(bucket, key):
    import boto3, joblib
    print(f"[INFO] Loading model from S3: s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp.seek(0)
        return joblib.load(tmp)


def load_model():
    source        = os.environ.get("MODEL_SOURCE", "auto").lower()
    tracking_uri  = os.environ.get("MLFLOW_TRACKING_URI", "")
    model_name    = os.environ.get("MODEL_NAME", "credit-card-approval")
    model_version = os.environ.get("MODEL_VERSION", "")
    local_path    = os.environ.get("MODEL_LOCAL_PATH", "")
    s3_bucket     = os.environ.get("S3_BUCKET", "")
    s3_key        = os.environ.get("MODEL_PKL", "")

    if source == "mlflow" or (source == "auto" and tracking_uri and model_name):
        return load_model_from_mlflow(tracking_uri, model_name, model_version)
    if source == "local" or (source == "auto" and local_path):
        return load_model_from_local(local_path)
    if source == "s3" or (source == "auto" and s3_bucket and s3_key):
        return load_model_from_s3(s3_bucket, s3_key)

    raise RuntimeError(
        "No model source configured. Set MLFLOW_TRACKING_URI, "
        "MODEL_LOCAL_PATH, or S3_BUCKET + MODEL_PKL."
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
