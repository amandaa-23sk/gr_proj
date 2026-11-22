import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from agentic_rag.rag.ingestion_clean import ingest_and_clean, DEFAULT_PDF_PATH

# Default database configurations
DEFAULT_PERSIST_DIR = os.path.join(PROJECT_ROOT, "agentic_rag", "rag", "vector_db", "chroma_pv_db")
DEFAULT_COLLECTION_NAME = "pv_failure_fact_sheets"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embedding_function(
    api_key: Optional[str] = None,
    model: str = DEFAULT_EMBEDDING_MODEL
) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI embeddings model using the Google API key.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Google API Key is required for embeddings. Please ensure GOOGLE_API_KEY is configured in your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key
    )


def build_or_load_vectorstore(
    pdf_path: str = DEFAULT_PDF_PATH,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    force_rebuild: bool = False,
    batch_size: int = 35,
    delay_between_batches: float = 25.0,
    api_key: Optional[str] = None
) -> Chroma:
    """
    Builds a persistent Chroma vector database from the ingested PDF chunks,
    or loads the existing vector store from disk if already created.
    Implements deterministic IDs and rate-limit handling for Google API free/standard tier.

    Args:
        pdf_path (str): Absolute or relative path to the technical PDF file.
        persist_dir (str): Directory where Chroma DB persists vector indices and metadata.
        collection_name (str): Name of the Chroma collection.
        embedding_model (str): Google Generative AI embedding model identifier.
        force_rebuild (bool): If True, clears existing vectors and re-ingests the PDF.
        batch_size (int): Number of document chunks embedded per API request batch.
        delay_between_batches (float): Seconds to sleep between batches to stay under the 100 RPM quota.
        api_key (str, optional): Google API key. Defaults to GOOGLE_API_KEY from .env.

    Returns:
        Chroma: Persistent LangChain Chroma vector store instance.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embedding_function(api_key=api_key, model=embedding_model)

    # Initialize Chroma client
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Clean existing entries if force_rebuild
    if force_rebuild:
        current_ids = vector_store._collection.get().get("ids", [])
        if current_ids:
            print(f"[!] Purging {len(current_ids)} existing vectors from '{collection_name}' for clean rebuild...")
            vector_store._collection.delete(ids=current_ids)
            print("[OK] Existing vectors purged.")

    current_ids = vector_store._collection.get().get("ids", [])
    current_count = len(current_ids)

    # Load existing store if already populated and rebuild is not requested
    if current_count > 0 and not force_rebuild:
        print(f"[+] Loaded existing Chroma Vector Store from: {persist_dir} ({current_count} vectors)")
        return vector_store

    # Ingest document chunks only when building or rebuilding
    print(f"[+] Ingesting documents for Vector Store creation from: {pdf_path}")
    chunks = ingest_and_clean(pdf_path)
    total_chunks = len(chunks)
    existing_id_set = set(current_ids)

    # Filter out chunks already embedded
    pending_items = [
        (f"chunk_{i:04d}", chunk)
        for i, chunk in enumerate(chunks)
        if f"chunk_{i:04d}" not in existing_id_set
    ]

    if not pending_items:
        print(f"[SUCCESS] All {total_chunks} chunks are already indexed in Chroma DB at: {persist_dir}")
        return vector_store

    print(f"[+] Found {len(pending_items)} chunks pending embedding (out of {total_chunks} total).")
    print(f"[+] Indexing with batch size = {batch_size} (pacing with {delay_between_batches}s delay to respect 100 RPM limit)...")

    total_pending = len(pending_items)
    total_batches = (total_pending + batch_size - 1) // batch_size

    for b_idx in range(0, total_pending, batch_size):
        batch_slice = pending_items[b_idx:b_idx + batch_size]
        batch_ids = [item[0] for item in batch_slice]
        batch_docs = [item[1] for item in batch_slice]
        batch_num = (b_idx // batch_size) + 1

        print(f"  -> Indexing batch {batch_num}/{total_batches} ({len(batch_docs)} chunks: {batch_ids[0]}..{batch_ids[-1]})...")

        # Retry with exponential backoff on 429 quota exhaustion
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                print(f"    [OK] Batch {batch_num}/{total_batches} indexed successfully.")
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    wait_time = 50.0 + (attempt * 10.0)
                    print(f"    [!] Rate limit (429) encountered on batch {batch_num}. Waiting {wait_time:.0f}s before retry (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries:
                        raise e
                    print(f"    [!] Error on batch {batch_num}: {e}. Retrying in 5s...")
                    time.sleep(5)

        # Pause slightly before next batch to stay under 100 RPM quota
        if b_idx + batch_size < total_pending:
            time.sleep(delay_between_batches)

    final_count = vector_store._collection.count()
    print(f"[SUCCESS] Chroma Vector Store indexing complete!")
    print(f"[SUCCESS] Total persistent vectors in collection '{collection_name}': {final_count}")

    return vector_store


def get_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None
) -> Chroma:
    """
    Convenience accessor: Loads the persistent Chroma vector store, or builds it if absent.
    """
    return build_or_load_vectorstore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        force_rebuild=False,
        api_key=api_key
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Initializing PV Knowledge Base Chroma Vector Store")
    print("=" * 60)
    vs = build_or_load_vectorstore()
    print(f"Verified vector count: {vs._collection.count()}")
