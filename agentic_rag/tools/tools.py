import os
import pandas as pd
from langchain.tools import tool

from agentic_rag.tools.model_loader import (
    get_model,
    get_yolo_model,
    get_shap_explainer,
    DEFAULT_MODEL_PATH,
    DEFAULT_EXPECTED_POWER_MODEL,
    DEFAULT_YOLO_MODEL_PATH,
    FAULT_CLASSES,
    YOLO_DEFECT_CLASSES
)


# ---------------------------------------------------------------------------
# Telemetry Fault Diagnosis Tool
# ---------------------------------------------------------------------------

def diagnose_sensor_readings(
    i1: float,
    v1: float = None,
    t1: float = None,
    irr: float = None,
    p1: float = None,
    model_identifier: str = DEFAULT_MODEL_PATH
) -> dict:
    """
    Diagnoses the operating state and potential faults of a solar PV panel from sensor readings.

    Args:
        i1 (float or dict): Panel current in Amperes (I1), or a dictionary with sensor keys.
        v1 (float, optional): Panel voltage in Volts (V1).
        t1 (float, optional): Panel temperature in Celsius (T1).
        irr (float, optional): Solar irradiance in W/m² (Irr).
        p1 (float, optional): Panel power in Watts (P1). Calculated as i1 * v1 if None.
        model_identifier (str, optional): Model key (e.g. 'pv_xgboost_s1') or path to .pkl model.

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

    # Load model and expected features using model_loader
    model, features = get_model(model_identifier)

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


# LangChain Agent Tool instance
pv_diagnosis_tool = tool(diagnose_sensor_readings)
diagnose_sensor_readings.invoke = pv_diagnosis_tool.invoke


# ---------------------------------------------------------------------------
# Digital Twin Power Prediction Helper
# ---------------------------------------------------------------------------

def _predict_panel_power(irr: float, t1: float, model_identifier: str = DEFAULT_EXPECTED_POWER_MODEL) -> float:
    """
    Internal helper to predict healthy benchmark power for given Irr and T1 using the Digital Twin.
    """
    irr = float(irr)
    t1 = float(t1)

    if irr <= 0.0:
        return 0.0

    model, _ = get_model(model_identifier)

    features_dict = {
        "Irr": irr,
        "T1": t1,
        "Irr_T1_interaction": irr * t1,
        "Irr_sq": irr ** 2,
        "T1_sq": t1 ** 2,
        "Irr_squared": irr ** 2,
        "T1_squared": t1 ** 2,
        "Irr_temp_derated": irr * (1.0 - 0.0045 * (t1 - 25.0)),
        "Irr_per_Kelvin": irr / (t1 + 273.15),
    }

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = ["Irr", "T1", "Irr_T1_interaction", "Irr_squared", "T1_squared", "Irr_temp_derated", "Irr_per_Kelvin"]

    input_df = pd.DataFrame([features_dict])[expected_cols]
    predicted_p = float(model.predict(input_df)[0])
    return max(0.0, round(predicted_p, 2))


# ---------------------------------------------------------------------------
# Expected Power (Digital Twin) Tool
# ---------------------------------------------------------------------------

def calculate_expected_power(
    irr: float,
    t1: float,
    p1_actual: float = None,
    model_identifier: str = DEFAULT_EXPECTED_POWER_MODEL
) -> dict:
    """
    Calculates the expected (healthy benchmark) power generation of Photovoltaic Panel 1
    under given environmental conditions (solar irradiance and temperature).
    Optionally evaluates power deficit (residual loss) and performance ratio if actual power is provided.

    Args:
        irr (float or dict): Solar irradiance in W/m² (Irr).
        t1 (float, optional): Panel surface temperature in Celsius (T1).
        p1_actual (float, optional): Actual measured power in Watts (P1) to evaluate health deficit.
        model_identifier (str, optional): Model key (default: 'pv_expected_power_p1') or path to .pkl model.

    Returns:
        dict: Expected power (Watts), power deficit, performance ratio (PR%), and health diagnosis.
    """
    # Support dictionary input if passed directly as first argument
    if isinstance(irr, dict):
        readings = irr
        irr = readings.get("Irr", readings.get("irr"))
        t1 = readings.get("T1", readings.get("t1", t1))
        p1_actual = readings.get("P1", readings.get("p1", p1_actual))

    if irr is None or t1 is None:
        raise ValueError("Missing required inputs: Irradiance (Irr) and Temperature (T1) are required.")

    irr = float(irr)
    t1 = float(t1)

    expected_power = _predict_panel_power(irr, t1, model_identifier)

    response = {
        "expected_power_watts": expected_power,
        "inputs": {
            "irradiance_wm2": irr,
            "temperature_celsius": t1
        }
    }

    # If actual power is provided, compute residual power loss and performance ratio
    if p1_actual is not None:
        actual_power = max(0.0, float(p1_actual))
        deficit = max(0.0, round(expected_power - actual_power, 2))
        pr_pct = min(150.0, round((actual_power / max(0.1, expected_power)) * 100.0, 2))

        # Health assessment based on Performance Ratio
        if pr_pct >= 90.0:
            health_status = "Optimal / Healthy Generation"
        elif pr_pct >= 75.0:
            health_status = "Mild Degradation / Soiling Detected"
        elif pr_pct >= 35.0:
            health_status = "Significant Power Loss / Shading or Degradation Suspected"
        else:
            health_status = "Critical Generation Collapse / Open Circuit or Severe Fault"

        response.update({
            "actual_power_watts": round(actual_power, 2),
            "power_deficit_watts": deficit,
            "performance_ratio_percent": pr_pct,
            "health_assessment": health_status
        })

    return response


# LangChain Agent Tool instance
pv_expected_power_tool = tool(calculate_expected_power)
calculate_expected_power.invoke = pv_expected_power_tool.invoke


# ---------------------------------------------------------------------------
# What-If Scenario Tool (Simulation & Counterfactual Analysis)
# ---------------------------------------------------------------------------

def simulate_what_if_scenario(
    irr: float,
    t1: float = 25.0,
    baseline_irr: float = None,
    baseline_t1: float = None,
    baseline_power: float = None,
    model_identifier: str = DEFAULT_EXPECTED_POWER_MODEL
) -> dict:
    """
    Simulates a 'What-If' scenario to predict expected solar PV panel power output under hypothetical environmental conditions.

    Use this tool whenever the user asks hypothetical or 'what if' questions, such as:
    - 'If the irradiance is 200 and panel temp is 25, what will be the power?'
    - 'What if irradiance drops to 150 W/m²?'
    - 'What power would we get if temperature rises to 50°C at 900 W/m² irradiance?'
    - 'What if the sun comes out and irradiance increases from 200 to 800 W/m²?'

    Args:
        irr (float or dict): Hypothetical/target solar irradiance in W/m² (e.g. 200.0). Can also accept a dictionary.
        t1 (float, optional): Hypothetical/target panel temperature in Celsius (e.g. 25.0). Defaults to 25.0°C (STC).
        baseline_irr (float, optional): Baseline/current solar irradiance in W/m² for comparison.
        baseline_t1 (float, optional): Baseline/current panel temperature in Celsius. Defaults to 25.0°C if baseline_irr is set.
        baseline_power (float, optional): Baseline/current actual power in Watts to compare generation difference.
        model_identifier (str, optional): Key or path for the trained expected power model.

    Returns:
        dict: Detailed what-if simulation report including:
            - scenario_inputs: Evaluated hypothetical irradiance and temperature.
            - expected_power_watts: Predicted healthy power generation under the scenario.
            - sunlight_classification: Description of solar irradiance condition.
            - stc_benchmark_power_watts: Standard Test Conditions (1000 W/m², 25°C) benchmark power.
            - capacity_factor_pct: Percentage of full STC rated power.
            - temperature_effect: Estimated thermal derating impact relative to STC 25°C.
            - baseline_comparison: (Optional) Delta and percentage changes if baseline data was provided.
            - scenario_summary: Clear, user-friendly textual summary of the what-if simulation outcome.
    """
    # Support dictionary input if passed directly as first argument
    if isinstance(irr, dict):
        readings = irr
        irr = readings.get("Irr", readings.get("irr"))
        t1 = readings.get("T1", readings.get("t1", t1))
        baseline_irr = readings.get("baseline_irr", readings.get("current_irr", baseline_irr))
        baseline_t1 = readings.get("baseline_t1", readings.get("current_t1", baseline_t1))
        baseline_power = readings.get("baseline_power", readings.get("current_power", baseline_power))

    if irr is None:
        raise ValueError("Missing required input: Solar irradiance (irr) must be provided for scenario simulation.")

    irr = float(irr)
    t1 = float(t1) if t1 is not None else 25.0

    # 1. Predict hypothetical expected power
    simulated_power = _predict_panel_power(irr, t1, model_identifier)

    # 2. Benchmark against Standard Test Conditions (STC: 1000 W/m², 25°C)
    stc_power = _predict_panel_power(1000.0, 25.0, model_identifier)
    capacity_factor = round((simulated_power / max(0.1, stc_power)) * 100.0, 2)

    # 3. Sunlight classification
    if irr < 100.0:
        sunlight_class = "Very Low Sunlight (Dawn/Dusk or Heavy Storm)"
    elif irr < 300.0:
        sunlight_class = "Low Sunlight (Overcast / Heavy Cloud Cover)"
    elif irr < 600.0:
        sunlight_class = "Moderate Sunlight (Scattered Clouds / Diffuse Daylight)"
    elif irr < 900.0:
        sunlight_class = "High Sunlight (Bright Clear Daylight)"
    else:
        sunlight_class = "Peak Solar Irradiance (Direct Sun)"

    # 4. Temperature derating impact relative to 25°C STC
    temp_delta = round(t1 - 25.0, 2)
    thermal_derating_pct = round(-0.0045 * temp_delta * 100.0, 2)

    result = {
        "scenario_inputs": {
            "hypothetical_irradiance_wm2": irr,
            "hypothetical_temperature_celsius": t1
        },
        "expected_power_watts": simulated_power,
        "sunlight_classification": sunlight_class,
        "stc_benchmark_power_watts": stc_power,
        "capacity_factor_pct": capacity_factor,
        "temperature_effect": {
            "temp_diff_from_stc_celsius": temp_delta,
            "thermal_derating_pct": thermal_derating_pct
        }
    }

    # 5. Baseline comparison if baseline telemetry or power was provided
    summary_baseline_text = ""
    if baseline_power is not None or baseline_irr is not None:
        if baseline_power is not None:
            base_p = float(baseline_power)
        else:
            base_t1 = float(baseline_t1) if baseline_t1 is not None else 25.0
            base_p = _predict_panel_power(float(baseline_irr), base_t1, model_identifier)

        power_delta = round(simulated_power - base_p, 2)
        pct_change = round((power_delta / max(0.1, base_p)) * 100.0, 2)
        trend = "increase" if power_delta > 0 else ("decrease" if power_delta < 0 else "no change")

        comparison_info = {
            "baseline_power_watts": round(base_p, 2),
            "power_change_watts": power_delta,
            "percentage_change": pct_change,
            "trend": trend
        }
        if baseline_irr is not None:
            comparison_info["baseline_irradiance_wm2"] = float(baseline_irr)
        if baseline_t1 is not None:
            comparison_info["baseline_temperature_celsius"] = float(baseline_t1)

        result["baseline_comparison"] = comparison_info
        summary_baseline_text = (
            f" Compared to the baseline power of {base_p:.2f} W, this is a {trend} "
            f"of {abs(power_delta):.2f} W ({pct_change:+.1f}%)."
        )

    # 6. User-friendly summary message
    result["scenario_summary"] = (
        f"Under what-if conditions of {irr:.1f} W/m² irradiance and {t1:.1f}°C panel temperature, "
        f"the expected healthy power generation is {simulated_power:.2f} Watts "
        f"({capacity_factor:.1f}% of STC full-sun capacity). "
        f"Environmental condition: {sunlight_class}.{summary_baseline_text}"
    )

    return result


# LangChain Agent Tool instance
pv_what_if_tool = tool(simulate_what_if_scenario)
what_if_scenario_tool = pv_what_if_tool
simulate_what_if_scenario.invoke = pv_what_if_tool.invoke


# ---------------------------------------------------------------------------
# PV Knowledge Base Literature Retriever Tool
# ---------------------------------------------------------------------------

def search_pv_failure_literature(
    query: str,
    top_k: int = 4
) -> dict:
    """
    Searches the official IEA PVPS Task 13 Photovoltaic Failure Fact Sheets (PVFS)
    and reliability literature for technical degradation mechanisms, failure modes,
    root causes, safety risks, diagnostic methods, and mitigation guidelines.

    Use this tool whenever you need:
    - Technical explanations of PV failure modes (e.g. Potential-Induced Degradation (PID),
      bypass diode failure, cell cracks, hot spots, delamination, snail trails, glass breakage).
    - Root cause analysis and environmental triggers (thermal cycling, humidity, UV, mechanical loading).
    - Safety categorization (fire hazards, electrical shock, physical danger).
    - Detection and diagnostic inspection methods (electroluminescence, thermography, I-V tracing).
    - Recommended mitigation, maintenance protocols, and IEC standards.

    Args:
        query (str or dict): Technical question or keywords to search in the PV reliability literature.
        top_k (int, optional): Number of relevant literature excerpts to retrieve (default: 4).

    Returns:
        dict: Formatted context, citations with page numbers, and summary of retrieved literature.
    """
    from agentic_rag.rag.retriever import retrieve_pv_knowledge_formatted

    # Support dictionary input if passed directly as first argument (common in LangChain agent tool calls)
    if isinstance(query, dict):
        args = query
        query = args.get("query", args.get("input", ""))
        top_k = args.get("top_k", top_k)

    if not query or not str(query).strip():
        raise ValueError("Missing search query. Please provide a technical question or search terms.")

    query_str = str(query).strip()
    top_k = int(top_k) if top_k else 4

    retrieval_res = retrieve_pv_knowledge_formatted(query=query_str, top_k=top_k)

    return {
        "query": query_str,
        "results_found": retrieval_res["retrieved_count"],
        "context": retrieval_res["formatted_context"],
        "citations": retrieval_res["citations"],
        "summary": (
            f"Retrieved {retrieval_res['retrieved_count']} relevant technical section(s) from "
            f"IEA-PVPS Task 13 Failure Fact Sheets for query: '{query_str}'."
        )
    }


# LangChain Agent Tool instances
pv_knowledge_retriever_tool = tool(search_pv_failure_literature)
pv_retriever_tool = pv_knowledge_retriever_tool
search_pv_failure_literature.invoke = pv_knowledge_retriever_tool.invoke


# ---------------------------------------------------------------------------
# Database Telemetry Retrieval & Natural Language Explanation Tool
# ---------------------------------------------------------------------------

def get_latest_pv_reading(limit: int = 1) -> dict:
    """
    Fetches the latest real-time sensor reading(s) from the PostgreSQL database (solar_db),
    evaluates actual generation vs Digital Twin expected power, verifies or infers the
    fault classification label, and generates a detailed natural language explanation.

    Use this tool whenever the user or agent asks:
    - 'What is the current/latest reading of the solar panel?'
    - 'What is the panel status right now from the database?'
    - 'Explain the current sensor readings and label.'
    - 'Is the panel generating expected power right now?'
    - 'Check the database for the latest telemetry.'

    Args:
        limit (int or dict, optional): Number of latest readings to fetch (default: 1).

    Returns:
        dict: Pure structured data containing timestamp, telemetry, power analysis,
              and diagnostic label for the LLM agent to interpret.
    """
    # Support dictionary argument if passed by LangChain Agent executor
    if isinstance(limit, dict):
        limit = limit.get("limit", 1)

    try:
        limit = int(limit) if limit else 1
    except (TypeError, ValueError):
        limit = 1

    from database.connection import fetch_latest_readings

    try:
        rows = fetch_latest_readings(limit=limit)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to query database: {e}"
        }

    if not rows:
        return {
            "status": "empty",
            "message": "The database 'solar_db' contains no telemetry records yet."
        }

    latest = rows[0]
    ts_val = latest.get("timestamp")
    timestamp_str = ts_val.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts_val, "strftime") else str(ts_val)

    temp = float(latest["temp"])
    irr = float(latest["irr"])
    v = float(latest["v"])
    c = float(latest["c"])

    # 1. Actual Power (Watts)
    actual_power = latest.get("actual_power")
    if actual_power is None or actual_power == 0.0:
        actual_power = round(v * c, 2)
    else:
        actual_power = round(float(actual_power), 2)

    # 2. Expected Power & Digital Twin Metrics
    expected_power = latest.get("expected_power")
    power_deficit = latest.get("power_deficit")
    performance_ratio = latest.get("performance_ratio")

    if expected_power is None or performance_ratio is None:
        twin_data = calculate_expected_power(irr=irr, t1=temp, p1_actual=actual_power)
        expected_power = twin_data["expected_power_watts"]
        power_deficit = twin_data["power_deficit_watts"]
        performance_ratio = twin_data["performance_ratio_percent"]
    else:
        expected_power = round(float(expected_power), 2)
        power_deficit = round(float(power_deficit if power_deficit is not None else max(0.0, expected_power - actual_power)), 2)
        performance_ratio = round(float(performance_ratio), 2)

    # 3. Fault Classification & Label
    label = latest.get("label")
    confidence = latest.get("confidence")

    if not label or confidence is None:
        diag_data = diagnose_sensor_readings(i1=c, v1=v, t1=temp, irr=irr, p1=actual_power)
        label = diag_data["diagnosis"]
        confidence = diag_data["confidence"]
        health_status = diag_data["status"]
    else:
        label = str(label)
        confidence = round(float(confidence), 4)
        health_status = "healthy" if "normal" in label.lower() else "fault_detected"

    result = {
        "status": "success",
        "timestamp": timestamp_str,
        "telemetry": {
            "temperature_celsius": temp,
            "irradiance_wm2": irr,
            "voltage_volts": v,
            "current_amperes": c
        },
        "power_analysis": {
            "actual_power_watts": actual_power,
            "expected_power_watts": expected_power,
            "power_deficit_watts": power_deficit,
            "performance_ratio_percent": performance_ratio
        },
        "diagnosis": {
            "label": label,
            "confidence": confidence,
            "health_status": health_status
        }
    }

    if limit > 1:
        result["additional_records_count"] = len(rows)

    return result



# LangChain Agent Tool instances
pv_latest_reading_tool = tool(get_latest_pv_reading)
get_latest_reading_tool = pv_latest_reading_tool
get_latest_pv_reading.invoke = pv_latest_reading_tool.invoke


# ---------------------------------------------------------------------------
# Computer Vision PV Surface Defect Inspection Tool (YOLO)
# ---------------------------------------------------------------------------

def inspect_pv_panel_image(
    image_path: str,
    conf_threshold: float = 0.25,
    model_path: str = DEFAULT_YOLO_MODEL_PATH
) -> dict:
    """
    Performs visual defect inspection on a solar PV panel image using a trained YOLO vision classifier.
    Detects physical surface anomalies including cracked glass, soiling, bird drops, electrical burns, and snow.

    Args:
        image_path (str or dict): Path to the image file (JPG, PNG, JPEG) of the solar panel.
        conf_threshold (float, optional): Minimum confidence threshold (default: 0.25).
        model_path (str, optional): Path or key to the YOLO model weights.

    Returns:
        dict: Inspection report containing detected defect class, confidence, severity assessment,
              full class probability distribution, and technical summary.
    """
    # Support dictionary argument if passed by LangChain Agent executor
    if isinstance(image_path, dict):
        args = image_path
        image_path = args.get("image_path", args.get("image", args.get("path", "")))
        conf_threshold = args.get("conf_threshold", conf_threshold)
        model_path = args.get("model_path", model_path)

    if not image_path or not str(image_path).strip():
        raise ValueError("Missing required image path. Please provide a valid path to a solar panel image.")

    image_path_str = str(image_path).strip()

    # Resolve image path relative to workspace if relative
    if not os.path.isabs(image_path_str):
        from agentic_rag.tools.model_loader import BASE_DIR
        candidate = os.path.join(BASE_DIR, image_path_str)
        if os.path.exists(candidate):
            image_path_str = candidate
        else:
            image_path_str = os.path.abspath(image_path_str)

    if not os.path.exists(image_path_str):
        raise FileNotFoundError(f"Solar panel image file not found at: {image_path_str}")

    # Load cached YOLO model
    yolo_model = get_yolo_model(model_path)

    # Run inference
    results = yolo_model(image_path_str, verbose=False)
    if not results:
        raise RuntimeError(f"YOLO vision inspection produced no output for image: {image_path_str}")

    res = results[0]
    probs = res.probs

    top1_idx = int(probs.top1)
    top1_class = res.names[top1_idx]
    top1_conf = float(probs.top1conf)

    # Severity and risk categorization based on defect type
    severity_mapping = {
        "Physical-Damage": "Critical - Mechanical cell fracture or shattered front glass; severe moisture ingress and electrical shock hazard.",
        "Electrical-damage": "Critical - Visible burn marks, junction box or busbar discoloration; severe thermal runaway and fire hazard.",
        "Bird-drop": "High - Localized shading obstruction creating reverse-bias hotspot hazard and localized heating.",
        "Dusty": "Medium - Uniform surface soiling layer causing optical attenuation and power degradation.",
        "Snow-Covered": "Medium/High - Extensive snow layer blocking sunlight absorption; clearing required.",
        "Clean": "Optimal - Panel surface is pristine with no visible physical anomalies or soiling."
    }

    defect_severity = severity_mapping.get(top1_class, "Unknown defect severity")
    is_defective = top1_class != "Clean"

    all_probabilities = {
        res.names[i]: round(float(probs.data[i]), 4)
        for i in range(len(res.names))
    }
    # Sort probabilities descending
    sorted_probs = dict(sorted(all_probabilities.items(), key=lambda item: item[1], reverse=True))

    summary = (
        f"Visual inspection of panel image '{os.path.basename(image_path_str)}' classified as "
        f"'{top1_class}' with {top1_conf:.2%} confidence. "
        f"Condition: {'Defect Detected' if is_defective else 'Healthy/Clean'}. Severity: {defect_severity}"
    )

    return {
        "status": "defect_detected" if is_defective else "healthy",
        "image_path": image_path_str,
        "image_filename": os.path.basename(image_path_str),
        "visual_defect": top1_class,
        "is_defective": is_defective,
        "confidence": round(top1_conf, 4),
        "defect_severity": defect_severity,
        "all_probabilities": sorted_probs,
        "inspection_summary": summary
    }


# LangChain Agent Tool instance
pv_visual_inspection_tool = tool(inspect_pv_panel_image)
inspect_pv_panel_image.invoke = pv_visual_inspection_tool.invoke


# ---------------------------------------------------------------------------
# Explainable AI (XAI) Root-Cause Explanation Tool (SHAP)
# ---------------------------------------------------------------------------

def explain_pv_fault_diagnosis(
    i1: float,
    v1: float = None,
    t1: float = None,
    irr: float = None,
    p1: float = None,
    model_identifier: str = DEFAULT_MODEL_PATH
) -> dict:
    """
    Explains the root-cause reasoning of the XGBoost PV fault diagnosis using SHAP (SHapley Additive exPlanations).
    Calculates exact game-theoretic feature attributions showing which telemetry parameters
    (Voltage, Current, Irradiance, Temperature, Power) pushed the model toward the diagnosed fault.

    Use this tool whenever:
    - The user asks 'Why was this fault diagnosed?' or 'Explain the root cause of this classification.'
    - You need to justify why a state is Short Circuit, Open Circuit, Shading, Degradation, or Normal.
    - Transparent feature contribution metrics and engineering explanations are required.

    Args:
        i1 (float or dict): Panel current in Amperes (I1), or a dictionary with sensor keys.
        v1 (float, optional): Panel voltage in Volts (V1).
        t1 (float, optional): Panel surface temperature in Celsius (T1).
        irr (float, optional): Solar irradiance in W/m² (Irr).
        p1 (float, optional): Panel power in Watts (P1). Calculated as i1 * v1 if None.
        model_identifier (str, optional): Model key or path to .pkl model.

    Returns:
        dict: Detailed XAI explanation report including diagnosed fault, confidence, feature attribution scores,
              dominant driver ranking, and a natural language explanation narrative.
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

    i1 = float(i1)
    v1 = float(v1)
    t1 = float(t1)
    irr = float(irr)

    if p1 is None:
        p1 = round(i1 * v1, 2)
    else:
        p1 = float(p1)

    model, features = get_model(model_identifier)

    telemetry = {
        "I1": i1,
        "V1": v1,
        "T1": t1,
        "P1": p1,
        "Irr": irr
    }
    input_df = pd.DataFrame([telemetry])[features]

    # Model inference
    probabilities = model.predict_proba(input_df)[0]
    pred_idx = int(model.predict(input_df)[0])
    diagnosed_class = FAULT_CLASSES.get(pred_idx, f"Class {pred_idx}")
    confidence = float(probabilities[pred_idx])

    # Compute SHAP values using cached TreeExplainer
    explainer = get_shap_explainer(model)
    shap_vals = explainer(input_df)

    # Extract SHAP contributions for the diagnosed class
    if len(shap_vals.values.shape) == 3:
        class_shaps = shap_vals.values[0, :, pred_idx]
    else:
        class_shaps = shap_vals.values[0, :]

    feature_attributions = {}
    for f_idx, f_name in enumerate(features):
        feature_attributions[f_name] = round(float(class_shaps[f_idx]), 4)

    # Sort features by absolute SHAP impact
    sorted_features = sorted(feature_attributions.items(), key=lambda item: abs(item[1]), reverse=True)

    # Identify primary drivers (positive contribution pushing toward this class)
    positive_drivers = [(f, val) for f, val in sorted_features if val > 0]

    top_feature, top_shap = sorted_features[0]

    # Calculate percentage contribution among positive drivers
    total_pos = sum(val for _, val in positive_drivers)
    driver_breakdown = []
    for f, val in positive_drivers:
        pct = round((val / max(0.0001, total_pos)) * 100.0, 1) if total_pos > 0 else 0.0
        driver_breakdown.append({
            "feature": f,
            "sensor_value": telemetry[f],
            "shap_value": val,
            "contribution_pct": pct,
            "impact": "Primary driver pushing towards fault"
        })

    # Generate engineering explanation narrative based on physics and SHAP drivers
    if diagnosed_class == "Short Circuit":
        narrative = (
            f"Diagnosed as 'Short Circuit' with {confidence:.2%} confidence. "
            f"The dominant root cause is near-zero Voltage (V1 = {v1:.2f} V, SHAP: {feature_attributions['V1']:+0.2f}), "
            f"which collapsed despite substantial irradiance ({irr:.1f} W/m²). "
            f"Current remains high (I1 = {i1:.2f} A, SHAP: {feature_attributions['I1']:+0.2f}), confirming closed circuit fault."
        )
    elif diagnosed_class == "Open Circuit":
        narrative = (
            f"Diagnosed as 'Open Circuit' with {confidence:.2%} confidence. "
            f"The primary root-cause driver is Voltage (V1 = {v1:.2f} V, SHAP: {feature_attributions['V1']:+0.2f}) "
            f"floating at open-circuit potential while Current (I1 = {i1:.2f} A, SHAP: {feature_attributions['I1']:+0.2f}) "
            f"dropped to near zero despite high irradiance ({irr:.1f} W/m²)."
        )
    elif diagnosed_class == "Shading":
        narrative = (
            f"Diagnosed as 'Shading' with {confidence:.2%} confidence. "
            f"The primary driver is Current suppression (I1 = {i1:.2f} A, SHAP: {feature_attributions['I1']:+0.2f}), "
            f"showing disproportionate photocurrent deficit relative to ambient irradiance ({irr:.1f} W/m²)."
        )
    elif diagnosed_class == "Degradation":
        narrative = (
            f"Diagnosed as 'Degradation' with {confidence:.2%} confidence. "
            f"The model detected power output shortfall (P1 = {p1:.2f} W, SHAP: {feature_attributions['P1']:+0.2f}) "
            f"driven by series resistance increase and mild voltage/current derating."
        )
    else:
        narrative = (
            f"Diagnosed as 'Normal' healthy operation with {confidence:.2%} confidence. "
            f"Sensor readings (I1 = {i1:.2f} A, V1 = {v1:.2f} V, P1 = {p1:.2f} W) closely match expected PV curve dynamics "
            f"under {irr:.1f} W/m² irradiance and {t1:.1f}°C temperature."
        )

    return {
        "status": "success",
        "diagnosed_class": diagnosed_class,
        "class_id": pred_idx,
        "confidence": round(confidence, 4),
        "telemetry_inputs": telemetry,
        "feature_attributions_shap": feature_attributions,
        "top_root_cause_driver": {
            "feature": top_feature,
            "value": telemetry[top_feature],
            "shap_value": top_shap
        },
        "positive_drivers": driver_breakdown,
        "explanation_narrative": narrative
    }


# LangChain Agent Tool instance
pv_xai_explainer_tool = tool(explain_pv_fault_diagnosis)
explain_pv_fault_diagnosis.invoke = pv_xai_explainer_tool.invoke



