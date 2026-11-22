import os 
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentic_rag.tools.tools import (
    diagnose_sensor_readings,
    pv_diagnosis_tool,
    calculate_expected_power,
    pv_expected_power_tool,
    simulate_what_if_scenario,
    pv_what_if_tool,
    inspect_pv_panel_image,
    pv_visual_inspection_tool,
    explain_pv_fault_diagnosis,
    pv_xai_explainer_tool
)

print("--- Telemetry Fault Diagnosis Tool ---")
# 1. Direct function call with keyword arguments
result = diagnose_sensor_readings(i1=5.80, v1=17.50, t1=35.0, irr=750.0)
print("1. Direct call:", result["diagnosis"], f"(Confidence: {result['confidence']})")

# 2. Direct function call with a dictionary
telemetry = {"I1": 0.01, "V1": 21.20, "T1": 42.0, "Irr": 910.0}
result = diagnose_sensor_readings(telemetry)
print("2. Dict call:  ", result["diagnosis"], f"(Status: {result['status']})")

# 3. In a LangChain / Agentic RAG workflow:
result = pv_diagnosis_tool.invoke({"i1": 5.80, "v1": 17.50, "t1": 35.0, "irr": 750.0})
print("3. Tool invoke:", result["diagnosis"], f"(Status: {result['status']})")

print("\n--- Digital Twin Expected Power Tool ---")
# 4. Direct call for expected power only
res_exp1 = calculate_expected_power(irr=850.0, t1=42.0)
print("4. Expected Power:", res_exp1["expected_power_watts"], "Watts")

# 5. Direct call with actual power (evaluates deficit & health)
res_exp2 = calculate_expected_power(irr=850.0, t1=42.0, p1_actual=38.0)
print("5. With Actual P1: Expected =", res_exp2["expected_power_watts"], "W | Actual =", res_exp2["actual_power_watts"], "W")
print("   Power Deficit:", res_exp2["power_deficit_watts"], "W | PR:", f"{res_exp2['performance_ratio_percent']}%")
print("   Health Assessment:", res_exp2["health_assessment"])

# 6. Tool invoke via LangChain interface
res_exp3 = pv_expected_power_tool.invoke({"irr": 600.0, "t1": 38.0, "p1_actual": 53.0})
print("6. LangChain Tool Invoke:", res_exp3["health_assessment"], f"(Expected: {res_exp3['expected_power_watts']} W)")

print("\n--- What-If Scenario Simulation Tool ---")
# 7. Direct call for what-if scenario (e.g. Irr = 200, T1 = 25)
res_whatif1 = simulate_what_if_scenario(irr=200.0, t1=25.0)
print("7. What-If (Irr=200, T1=25): Expected Power =", res_whatif1["expected_power_watts"], "W")
print("   Condition:", res_whatif1["sunlight_classification"])
print("   Summary:", res_whatif1["scenario_summary"])

# 8. LangChain Tool Invoke for what-if scenario
res_whatif2 = pv_what_if_tool.invoke({"irr": 200.0, "t1": 25.0})
print("\n8. LangChain What-If Tool Invoke:")
print("   Expected Power:", res_whatif2["expected_power_watts"], "W")
print("   STC Benchmark:", res_whatif2["stc_benchmark_power_watts"], "W")
print("   Capacity Factor:", f"{res_whatif2['capacity_factor_pct']}%")

# 9. What-If Scenario with baseline comparison
res_whatif3 = simulate_what_if_scenario(irr=200.0, t1=25.0, baseline_power=75.0, baseline_irr=850.0, baseline_t1=40.0)
print("\n9. What-If with Baseline Comparison:")
print("   Summary:", res_whatif3["scenario_summary"])
print("   Baseline Comparison:", res_whatif3["baseline_comparison"])

print("\n--- Computer Vision Visual Inspection Tool (YOLO) ---")
# 10. Direct call with test image
sample_img = os.path.join(PROJECT_ROOT, "images", "Damaged-solar-panel-cracked-glass.jpg")
res_vision1 = inspect_pv_panel_image(sample_img)
print("10. Visual Defect:", res_vision1["visual_defect"], f"(Confidence: {res_vision1['confidence']:.2%})")
print("    Severity:", res_vision1["defect_severity"])
print("    Summary :", res_vision1["inspection_summary"])

# 11. LangChain Tool Invoke for Visual Inspection
res_vision2 = pv_visual_inspection_tool.invoke({"image_path": sample_img})
print("\n11. LangChain Vision Invoke:", res_vision2["visual_defect"], f"(Status: {res_vision2['status']})")

print("\n--- Explainable AI Root-Cause Explainer Tool (SHAP) ---")
# 12. Direct call for Short Circuit telemetry
sc_telemetry = {"I1": 7.85, "V1": 0.05, "T1": 45.0, "Irr": 850.0}
res_xai1 = explain_pv_fault_diagnosis(sc_telemetry)
print("12. XAI Diagnosis:", res_xai1["diagnosed_class"], f"(Confidence: {res_xai1['confidence']:.2%})")
print("    Top Driver  :", res_xai1["top_root_cause_driver"])
print("    Narrative   :", res_xai1["explanation_narrative"])

# 13. LangChain Tool Invoke for Open Circuit telemetry
oc_input = {"i1": 0.01, "v1": 21.20, "t1": 42.0, "irr": 910.0}
res_xai2 = pv_xai_explainer_tool.invoke(oc_input)
print("\n13. LangChain XAI Invoke:", res_xai2["diagnosed_class"], f"(Confidence: {res_xai2['confidence']:.2%})")
print("    Narrative   :", res_xai2["explanation_narrative"])

