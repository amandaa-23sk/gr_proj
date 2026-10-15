import os
from typing import Any, Dict, List, Tuple
import joblib

# Base workspace directory and models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default models registry
AVAILABLE_MODELS = {
    "pv_xgboost_s1": os.path.join(MODELS_DIR, "pv_xgboost_s1.pkl"),
    "pv_xgboost_s2": os.path.join(MODELS_DIR, "pv_xgboost_s2.pkl"),
    "pv_random_forest_s1": os.path.join(MODELS_DIR, "pv_random_forest_s1.pkl"),
    "pv_random_forest_s2": os.path.join(MODELS_DIR, "pv_random_forest_s2.pkl"),
    "pv_digital_twin": os.path.join(MODELS_DIR, "pv_digital_twin_p1_xgb.pkl"),
}

# Default model path for Panel 1 XGBoost classifier
DEFAULT_MODEL_PATH = AVAILABLE_MODELS["pv_xgboost_s1"]

# Fault classification classes mapping
FAULT_CLASSES = {
    0: "Normal",
    1: "Short Circuit",
    2: "Open Circuit",
    3: "Degradation",
    4: "Shading"
}

# Cache to store loaded models: {path_or_key: (model_obj, features_list)}
_model_cache: Dict[str, Tuple[Any, List[str]]] = {}


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
