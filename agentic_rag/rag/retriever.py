import os
import sys
from typing import List, Optional, Dict, Any
# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from agentic_rag.rag.vectorstore import (
    get_vectorstore,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL
)


def get_pv_retriever(
    search_type: str = "similarity",
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    filter_dict: Optional[dict] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
):
    """
    Returns a configured LangChain VectorStoreRetriever for the PV knowledge base.

    Args:
        search_type (str): Retrieval algorithm: 'similarity', 'similarity_score_threshold', or 'mmr'.
        top_k (int): Number of top relevant documents to retrieve.
        score_threshold (float, optional): Minimum similarity score threshold (when search_type='similarity_score_threshold').
        filter_dict (dict, optional): Metadata filter conditions for Chroma DB.
        persist_dir (str): Path to the persistent Chroma DB storage.
        collection_name (str): Name of the Chroma collection.
        embedding_model (str): Google Generative AI embedding model.

    Returns:
        VectorStoreRetriever: Configured LangChain retriever.
    """
    vector_store = get_vectorstore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

    search_kwargs: Dict[str, Any] = {"k": top_k}
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    if filter_dict is not None:
        search_kwargs["filter"] = filter_dict

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )


def retrieve_pv_context(
    query: str,
    top_k: int = 4,
    search_type: str = "similarity",
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> List[Document]:
    """
    Directly retrieves relevant Document objects for a given technical query.

    Args:
        query (str): The search query or technical question.
        top_k (int): Number of documents to retrieve.
        search_type (str): 'similarity' or 'mmr'.
        persist_dir (str): Path to Chroma DB directory.
        collection_name (str): Chroma collection name.

    Returns:
        List[Document]: List of matching LangChain Document instances with content and metadata.
    """
    retriever = get_pv_retriever(
        search_type=search_type,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name
    )
    return retriever.invoke(query)


def retrieve_pv_knowledge_formatted(
    query: str,
    top_k: int = 4,
    search_type: str = "similarity",
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> Dict[str, Any]:
    """
    Retrieves and formats technical literature into a clean, structured dictionary
    with page citations and excerpt text suitable for LLMs and agent tool responses.

    Args:
        query (str): Technical question or query keywords.
        top_k (int): Number of passages to return.
        search_type (str): Search type ('similarity' or 'mmr').
        persist_dir (str): Path to Chroma DB directory.
        collection_name (str): Chroma collection name.

    Returns:
        Dict[str, Any]: Structured retrieval results including citations, formatted context, and raw documents.
    """
    docs = retrieve_pv_context(
        query=query,
        top_k=top_k,
        search_type=search_type,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    if not docs:
        return {
            "query": query,
            "retrieved_count": 0,
            "formatted_context": "No relevant literature found in the PV Failure Fact Sheets for this query.",
            "citations": [],
            "documents": []
        }

    formatted_parts = []
    citations = []
    doc_dicts = []

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "IEA-PVPS Task 13")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content.strip()

        citation = {
            "index": idx,
            "source": source,
            "page": page,
            "char_count": len(snippet)
        }
        citations.append(citation)

        formatted_parts.append(
            f"--- [Reference #{idx} | Source: {source} | Page: {page}] ---\n{snippet}"
        )

        doc_dicts.append({
            "index": idx,
            "metadata": doc.metadata,
            "content": snippet
        })

    return {
        "query": query,
        "retrieved_count": len(docs),
        "formatted_context": "\n\n".join(formatted_parts),
        "citations": citations,
        "documents": doc_dicts
    }


if __name__ == "__main__":
    test_query = "What causes Potential-Induced Degradation (PID) and how is it detected?"
    print(f"Testing retrieval for: '{test_query}'...")
    results = retrieve_pv_knowledge_formatted(test_query, top_k=2)
    print(f"Found {results['retrieved_count']} references:")
    for cit in results["citations"]:
        print(f" - Page {cit['page']}: {cit['source']}")
    print("\nExcerpt:\n", results["formatted_context"][:500] + "...")
