# model_governance.rego
#
# Harness Governance — ML Model Quality and Fairness Policy
#
# Evaluated by the "Governance Policy Check" stage in the Harness MLOps pipeline.
# Any rule that fires populates the `deny` set, causing the pipeline to halt
# and the model to remain in PendingManualApproval status.
#
# Policy owners: ML Platform / Risk & Compliance teams
# Versioned alongside the pipeline — changes require PR review.
#
# Usage (via check_governance.py):
#   opa eval --format json \
#            --input model_metrics.json \
#            --data policies/model_governance.rego \
#            "data.mlops.model_governance.deny"

package mlops.model_governance

# ── Thresholds ────────────────────────────────────────────────────────────────
# Minimum acceptable performance on the held-out test set.
# Align these with the thresholds in evaluate_model.py.

minimum_accuracy  := 0.80
minimum_precision := 0.75
minimum_recall    := 0.75
minimum_f1        := 0.75

# Maximum allowed approval-rate gap between demographic groups.
# Exceeding this threshold indicates the model may discriminate.
maximum_fairness_gap := 0.20

# ── Deny rules ────────────────────────────────────────────────────────────────

deny contains msg if {
    input.test_accuracy < minimum_accuracy
    msg := sprintf(
        "ACCURACY BELOW THRESHOLD: %.4f < %.2f — model is not accurate enough for production",
        [input.test_accuracy, minimum_accuracy]
    )
}

deny contains msg if {
    input.test_precision < minimum_precision
    msg := sprintf(
        "PRECISION BELOW THRESHOLD: %.4f < %.2f — too many false approvals risk financial loss",
        [input.test_precision, minimum_precision]
    )
}

deny contains msg if {
    input.test_recall < minimum_recall
    msg := sprintf(
        "RECALL BELOW THRESHOLD: %.4f < %.2f — model is missing too many creditworthy applicants",
        [input.test_recall, minimum_recall]
    )
}

deny contains msg if {
    input.test_f1 < minimum_f1
    msg := sprintf(
        "F1 BELOW THRESHOLD: %.4f < %.2f — overall model quality is insufficient",
        [input.test_f1, minimum_f1]
    )
}

deny contains msg if {
    input.fairness_gap >= maximum_fairness_gap
    msg := sprintf(
        "FAIRNESS VIOLATION: approval-rate gap %.4f >= %.2f — model shows potential demographic bias",
        [input.fairness_gap, maximum_fairness_gap]
    )
}

# ── Allow rule ────────────────────────────────────────────────────────────────

allow if {
    count(deny) == 0
}
