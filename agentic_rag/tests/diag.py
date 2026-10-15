import os 
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentic_rag.tools.tools import diagnose_sensor_readings, pv_diagnosis_tool

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
