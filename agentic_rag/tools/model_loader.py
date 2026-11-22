import os
from typing import Any, Dict, List, Tuple
import joblib

# Base workspace directory and models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default models registry
AVAILABLE_MODELS = {
    "pv_xgboost_s1": os.path.join(MODELS_DIR, "pv_xgboost_s1.pkl"),
    "pv_digital_twin": os.path.join(MODELS_DIR, "pv_digital_twin_p1_xgb.pkl"),
    "pv_expected_power_p1": os.path.join(MODELS_DIR, "pv_expected_power_p1_xgb.pkl")
}

# Default model path for Panel 1 Expected Power (Digital Twin)
DEFAULT_EXPECTED_POWER_MODEL = AVAILABLE_MODELS["pv_expected_power_p1"]

# Default model path for Panel 1 XGBoost classifier
DEFAULT_MODEL_PATH = AVAILABLE_MODELS["pv_xgboost_s1"]

# Default model path for YOLO PV Defect Classification
DEFAULT_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best (3).pt")

# Fault classification classes mapping
FAULT_CLASSES = {
    0: "Normal",
    1: "Short Circuit",
    2: "Open Circuit",
    3: "Degradation",
    4: "Shading"
}

# YOLO visual defect classes
YOLO_DEFECT_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

# Cache to store loaded models: {path_or_key: (model_obj, features_list)}
_model_cache: Dict[str, Tuple[Any, List[str]]] = {}
_yolo_model_cache: Dict[str, Any] = {}
_shap_explainer_cache: Dict[int, Any] = {}


def get_model(model_identifier: str = DEFAULT_MODEL_PATH) -> Tuple[Any, List[str]]:
    """
    Loads and caches any pre-trained model checkpoint.
    Supports predefined model keys (e.g. 'pv_xgboost_s1') or explicit file paths.

    Args:
        model_identifier (str): Key from AVAILABLE_MODELS or full/relative path to a .pkl model.

    Returns:
        tuple: (model_object, features_list)
    """
    # Resolve model path from registry key or file path
    if model_identifier in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_identifier]
    elif os.path.isabs(model_identifier):
        model_path = model_identifier
    else:
        # Check if file exists relative to models directory or current path
        candidate_path = os.path.join(MODELS_DIR, model_identifier)
        model_path = candidate_path if os.path.exists(candidate_path) else os.path.abspath(model_identifier)

    # Return cached model if already loaded in memory
    if model_path in _model_cache:
        return _model_cache[model_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    pkg = joblib.load(model_path)

    # Unpack model and feature names if saved as a package dict
    if isinstance(pkg, dict):
        model = pkg.get("model", pkg)
        features = pkg.get("features", ["I1", "V1", "T1", "P1", "Irr"])
    else:
        model = pkg
        features = ["I1", "V1", "T1", "P1", "Irr"]

    _model_cache[model_path] = (model, features)
    return model, features


def get_yolo_model(model_identifier: str = DEFAULT_YOLO_MODEL_PATH) -> Any:
    """
    Loads and caches the pre-trained YOLO PV panel surface defect classification model.

    Args:
        model_identifier (str): Path or key to the .pt model weights.

    Returns:
        YOLO: Cached Ultralytics YOLO model instance.
    """
    global _yolo_model_cache
    if "YOLO_CONFIG_DIR" not in os.environ:
        os.environ["YOLO_CONFIG_DIR"] = os.path.join(BASE_DIR, ".ultralytics")

    if model_identifier in _yolo_model_cache:
        return _yolo_model_cache[model_identifier]

    if os.path.isabs(model_identifier):
        model_path = model_identifier
    else:
        candidate_path = os.path.join(MODELS_DIR, model_identifier)
        model_path = candidate_path if os.path.exists(candidate_path) else os.path.abspath(model_identifier)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")

    from ultralytics import YOLO
    yolo_model = YOLO(model_path)
    _yolo_model_cache[model_identifier] = yolo_model
    return yolo_model


def get_shap_explainer(model: Any = None) -> Any:
    """
    Loads and caches a SHAP TreeExplainer for the XGBoost model with automatic
    XGBoost 3.x multiclass ubjson compatibility.

    Args:
        model: Pre-trained XGBoost model. If None, loads DEFAULT_MODEL_PATH.

    Returns:
        TreeExplainer: Cached SHAP TreeExplainer instance.
    """
    global _shap_explainer_cache
    if model is None:
        model, _ = get_model(DEFAULT_MODEL_PATH)

    model_id = id(model)
    if model_id in _shap_explainer_cache:
        return _shap_explainer_cache[model_id]

    import shap.explainers._tree as tr
    # Apply monkey patch to decode_ubjson_buffer if needed for XGBoost 3.x string array base_score
    if not getattr(tr, "_xgboost3_patched", False):
        orig_decode = tr.decode_ubjson_buffer

        def patched_decode(fd):
            res = orig_decode(fd)
            try:
                lmp = res.get("learner", {}).get("learner_model_param", {})
                raw = lmp.get("base_score")
                if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
                    vals = [float(x.strip()) for x in raw[1:-1].split(",")]
                    lmp["base_score"] = vals[0]
            except Exception:
                pass
            return res

        tr.decode_ubjson_buffer = patched_decode
        tr._xgboost3_patched = True

    explainer = tr.TreeExplainer(model)
    _shap_explainer_cache[model_id] = explainer
    return explainer

