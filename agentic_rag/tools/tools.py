import os
import joblib
import pandas as pd

from langchain.tools import tool


# Model path for PV panel 1 fault classification model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "pv_xgboost_s1.pkl")

# Operational class labels
FAULT_CLASSES = {
    0: "Normal",
    1: "Short Circuit",
    2: "Open Circuit",
    3: "Degradation",
    4: "Shading"
}

_model = None
_features = None


def _get_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and cache the trained XGBoost model package."""
    global _model, _features
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        pkg = joblib.load(model_path)
        if isinstance(pkg, dict):
            _model = pkg.get("model", pkg)
            _features = pkg.get("features", ["I1", "V1", "T1", "P1", "Irr"])
        else:
            _model = pkg
            _features = ["I1", "V1", "T1", "P1", "Irr"]
    return _model, _features


def diagnose_sensor_readings(
    i1: float,
    v1: float = None,
    t1: float = None,
    irr: float = None,
    p1: float = None,
    model_path: str = DEFAULT_MODEL_PATH
) -> dict:
    """
    Diagnoses the operating state and potential faults of a solar PV panel from sensor readings.

    Args:
        i1 (float or dict): Panel current in Amperes (I1), or a dictionary with sensor keys.
        v1 (float, optional): Panel voltage in Volts (V1).
        t1 (float, optional): Panel temperature in Celsius (T1).
        irr (float, optional): Solar irradiance in W/m² (Irr).
        p1 (float, optional): Panel power in Watts (P1). Calculated as i1 * v1 if None.
        model_path (str, optional): Path to the trained pv_xgboost_s1.pkl checkpoint.

    Returns:
        dict: Diagnostic result with status, diagnosis, confidence, and telemetry.
    """
    # Support dictionary input if passed directly as first argument
    if isinstance(i1, dict):
        readings = i1
        i1 = readings.get("I1", readings.get("i1"))
        v1 = readings.get("V1", readings.get("v1"))
        t1 = readings.get("T1", readings.get("t1"))
        irr = readings.get("Irr", readings.get("irr"))
        p1 = readings.get("P1", readings.get("p1"))

    if i1 is None or v1 is None or t1 is None or irr is None:
        raise ValueError("Missing required sensor readings. Please provide I1, V1, T1, and Irr.")

    # Calculate power P1 = I1 * V1 if not provided
    if p1 is None:
        p1 = float(i1) * float(v1)

    model, features = _get_model(model_path)

    # Prepare DataFrame matching model features
    telemetry = {
        "I1": float(i1),
        "V1": float(v1),
        "T1": float(t1),
        "P1": float(p1),
        "Irr": float(irr)
    }
    input_df = pd.DataFrame([telemetry])[features]

    # Predict probabilities and fault class
    probabilities = model.predict_proba(input_df)[0]
    pred_idx = int(model.predict(input_df)[0])
    diagnosis = FAULT_CLASSES.get(pred_idx, f"Class {pred_idx}")
    confidence = float(probabilities[pred_idx])

    return {
        "status": "healthy" if pred_idx == 0 else "fault_detected",
        "diagnosis": diagnosis,
        "class_id": pred_idx,
        "confidence": round(confidence, 4),
        "sensor_readings": telemetry,
        "probabilities": {
            FAULT_CLASSES.get(idx, f"Class {idx}"): round(float(prob), 4)
            for idx, prob in enumerate(probabilities)
        }
    }


# Export LangChain tool instance for Agentic RAG workflows
try:
    pv_diagnosis_tool = tool(diagnose_sensor_readings)
    diagnose_sensor_readings.invoke = pv_diagnosis_tool.invoke
except Exception:
    pv_diagnosis_tool = diagnose_sensor_readings
