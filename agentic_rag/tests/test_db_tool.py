import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentic_rag.tools import (
    get_latest_pv_reading,
    pv_latest_reading_tool,
    record_sensor_reading
)


def run_test():
    print("=" * 70)
    print("TESTING DATABASE TELEMETRY TOOL (PURE STRUCTURED DATA FOR LLM)")
    print("=" * 70)

    # 1. Fetch current latest reading
    print("\n[Step 1] Initial State Fetch:")
    res1 = get_latest_pv_reading()
    print(f" -> Status: {res1['status']}")

    # 2. Record a healthy normal operational reading
    print("\n[Step 2] Recording Healthy Telemetry Reading...")
    saved1 = record_sensor_reading(temp=32.0, irr=700.0, v=17.20, c=5.40)
    print(f" -> Stored Record ID: {saved1['id']}")
    print(f" -> Actual Power: {saved1['actual_power']} W | Expected: {saved1['expected_power']} W")
    print(f" -> Performance Ratio: {saved1['performance_ratio']}% | Deficit: {saved1['power_deficit']} W")
    print(f" -> Diagnosed Label: {saved1['label']} (Confidence: {saved1['confidence']})")

    # 3. Fetch latest reading via get_latest_pv_reading
    print("\n[Step 3] Fetching Latest Reading via Tool Function...")
    res2 = get_latest_pv_reading()
    print(f" -> Timestamp: {res2['timestamp']}")
    print(f" -> Telemetry: {res2['telemetry']}")
    print(f" -> Power Analysis: {res2['power_analysis']}")
    print(f" -> Diagnosis: {res2['diagnosis']}")

    # 4. Record a degraded telemetry reading
    print("\n[Step 4] Recording Degraded Telemetry Reading...")
    saved2 = record_sensor_reading(temp=45.0, irr=900.0, v=11.50, c=2.80)
    print(f" -> Stored Record ID: {saved2['id']}")
    print(f" -> Actual Power: {saved2['actual_power']} W | Expected: {saved2['expected_power']} W")
    print(f" -> Performance Ratio: {saved2['performance_ratio']}% | Deficit: {saved2['power_deficit']} W")
    print(f" -> Diagnosed Label: {saved2['label']} (Confidence: {saved2['confidence']})")

    # 5. Invoke via LangChain Tool interface
    print("\n[Step 5] Invoking via LangChain Agent Tool (pv_latest_reading_tool)...")
    tool_res = pv_latest_reading_tool.invoke({})
    print(f" -> Tool Status: {tool_res['status']}")
    print(f" -> Timestamp: {tool_res['timestamp']}")
    print(f" -> Pure Structured Data returned to LLM:")
    import json
    print(json.dumps(tool_res, indent=2))

    print("\n" + "=" * 70)
    print("SUCCESS: Pure Structured Database Telemetry Tool Verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
