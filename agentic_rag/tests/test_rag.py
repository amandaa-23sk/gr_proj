import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentic_rag.rag import (
    get_vectorstore,
    get_pv_retriever,
    retrieve_pv_knowledge_formatted
)
from agentic_rag.tools import (
    search_pv_failure_literature,
    pv_knowledge_retriever_tool,
    diagnose_sensor_readings
)


def run_rag_validation():
    print("=" * 70)
    print("PV SYSTEM AGENTIC RAG & RETRIEVAL VALIDATION SUITE")
    print("=" * 70)

    # 1. Validate Vector Store Persistence
    print("\n[Test 1] Validating Persistent Chroma Vector Store...")
    vs = get_vectorstore()
    vector_count = vs._collection.count()
    print(f" -> Successfully loaded Chroma DB with {vector_count} indexed chunks.")
    assert vector_count > 0, "Vector store should not be empty."

    # 2. Validate Direct Retrieval Functionality
    print("\n[Test 2] Testing Context Retrieval with Citations...")
    query = "What is Potential-Induced Degradation (PID) and how does it affect cell efficiency?"
    retrieval_res = retrieve_pv_knowledge_formatted(query=query, top_k=2)
    print(f" -> Query: '{retrieval_res['query']}'")
    print(f" -> Chunks Found: {retrieval_res['retrieved_count']}")
    for cit in retrieval_res["citations"]:
        print(f"    * Citation #{cit['index']}: Page {cit['page']} in {cit['source']} ({cit['char_count']} chars)")
    assert retrieval_res["retrieved_count"] > 0, "Expected at least 1 document chunk."

    # 3. Validate LangChain Retriever Tool Interface
    print("\n[Test 3] Testing LangChain Tool Invocation (pv_knowledge_retriever_tool)...")
    tool_input = {"query": "Bypass diode failure mechanisms and safety hazard categories", "top_k": 2}
    tool_res = pv_knowledge_retriever_tool.invoke(tool_input)
    print(f" -> Tool Response Summary: {tool_res['summary']}")
    print(f" -> Results Found: {tool_res['results_found']}")
    print(f" -> Citations: {tool_res['citations']}")
    print(" -> Context Snippet:")
    print("    " + tool_res['context'][:300].replace("\n", "\n    ") + "...")
    assert tool_res["results_found"] > 0, "Expected tool results."

    # 4. Multi-Modal Synergistic Workflow (Telemetry + RAG Knowledge)
    print("\n[Test 4] Demonstrating Joint Workflow: Telemetry Diagnosis + RAG Grounding...")
    # Simulate a degraded panel telemetry reading
    telemetry = {"I1": 4.10, "V1": 14.80, "T1": 46.0, "Irr": 820.0}
    diagnosis_output = diagnose_sensor_readings(telemetry)
    detected_state = diagnosis_output["diagnosis"]
    print(f" -> Telemetry Diagnostic Output: State = '{detected_state}' (Confidence: {diagnosis_output['confidence']})")

    # Retrieve domain knowledge for the diagnosed fault
    rag_query = f"Causes and field mitigation for PV {detected_state}"
    rag_output = search_pv_failure_literature(rag_query, top_k=1)
    print(f" -> Automated RAG Literature Search for: '{rag_query}'")
    print(f" -> Grounded Knowledge Citation: Page {rag_output['citations'][0]['page']} from {rag_output['citations'][0]['source']}")
    print(" -> Knowledge Excerpt:")
    print("    " + rag_output['context'][:300].replace("\n", "\n    ") + "...")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: Vector Store, Retrieval, and Agent Tools Verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_rag_validation()
