import os
import re
import sys
from typing import List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default target PDF path
DEFAULT_PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "agentic_rag",
    "data",
    "IEA-PVPS-T13-30-2025-PVFS-ANNEX-Degradation-and-Failure.pdf"
)


def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text by removing recurring headers, footers,
    page numbers, hyphenated line wraps, and normalizing whitespace.
    """
    if not text:
        return ""

    # 1. Remove recurring Task 13 document header/footer lines
    text = re.sub(
        r"Task 13 Reliability and Performance of Photovoltaic Systems[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 2. Fix broken hyphenated line breaks (e.g. "Fed-\neral" -> "Federal")
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # 3. Replace non-printable / unknown replacement characters
    text = text.replace("\ufffd", " ")

    # 4. Remove isolated page numbers on their own lines
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # 5. Normalize horizontal whitespaces (tabs, multiple spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Normalize multiple empty lines to max 2 newlines (paragraph boundary)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def load_and_clean_pdf(pdf_path: str = DEFAULT_PDF_PATH) -> List[Document]:
    """
    Loads all pages of the PDF, cleans the extracted text, and wraps
    valid content into LangChain Document objects with page metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = pypdf.PdfReader(pdf_path)
    documents: List[Document] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        # Skip completely blank or noise-only pages
        if not cleaned or len(cleaned) < 20:
            continue

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "page": idx,
                "total_pages": len(reader.pages)
            }
        )
        documents.append(doc)

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits clean documents into smaller semantic chunks for vector embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def ingest_and_clean(
    pdf_path: str = DEFAULT_PDF_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    End-to-end pipeline: Ingests PDF, cleans text page-by-page, and returns chunks.
    """
    print(f"[+] Ingesting and cleaning PDF: {pdf_path} ...")
    clean_docs = load_and_clean_pdf(pdf_path)
    print(f"[OK] Extracted {len(clean_docs)} clean pages.")

    chunks = chunk_documents(clean_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


if __name__ == "__main__":
    chunks = ingest_and_clean()
    if chunks:
        print("\n" + "=" * 60)
        print("Sample First Chunk:")
        print(f"Page: {chunks[0].metadata.get('page')}")
        print(f"Length: {len(chunks[0].page_content)} characters")
        print("-" * 60)
        print(chunks[0].page_content[:400] + "...")
        print("=" * 60)

