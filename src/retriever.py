"""
src/retriever.py — PDF Ingestion, Chunking, Embedding & ChromaDB Operations
RAG-Based Customer Support Assistant
"""

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Module Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vectordb")
COLLECTION_NAME    = "support_kb"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # Lightweight, fast, high-quality
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 100
TOP_K_RESULTS      = 4


# ─── 1. Load PDF ──────────────────────────────────────────────────────────────
def load_pdf(file_path: str):
    """
    Load a PDF file and return a list of LangChain Document objects.
    Each Document corresponds to one page of the PDF.
    """
    logger.info(f"Loading PDF: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("PDF appears to be empty or could not be parsed.")

    logger.info(f"Loaded {len(documents)} pages from PDF.")
    return documents


# ─── 2. Split Into Chunks ─────────────────────────────────────────────────────
def split_chunks(documents: list):
    """
    Split loaded documents into smaller, overlapping text chunks.
    Overlap ensures that context isn't lost at chunk boundaries.
    """
    logger.info("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],  # Smart split order
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created. Check if the PDF has extractable text.")

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


# ─── 3. Create Embedding Model ────────────────────────────────────────────────
def create_embeddings():
    """
    Initialise the HuggingFace sentence-transformer embedding model.
    Downloads locally on first run; cached afterwards.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},          # CPU-safe for all machines
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully.")
    return embeddings


# ─── 4. Store in ChromaDB ─────────────────────────────────────────────────────
def store_chromadb(chunks: list, embeddings):
    """
    Embed all chunks and persist them into a local ChromaDB vector store.
    If the collection already exists it will be overwritten.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB at: {CHROMA_PERSIST_DIR}")

    # Build the vector store (Chroma auto-persists with persist_directory)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    logger.info("ChromaDB vector store created and persisted successfully.")
    return vectordb


# ─── 5. Load Existing ChromaDB ────────────────────────────────────────────────
def load_chromadb(embeddings):
    """
    Load an existing ChromaDB vector store from disk.
    Used at query time to avoid re-embedding on every request.
    """
    logger.info("Loading existing ChromaDB vector store...")

    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    logger.info("ChromaDB loaded from disk.")
    return vectordb


# ─── 6. Retrieve Context ──────────────────────────────────────────────────────
def retrieve_context(query: str, vectordb) -> str:
    """
    Perform a semantic similarity search and return the top-K relevant chunks
    concatenated into a single context string for the LLM.
    """
    logger.info(f"Retrieving context for query: {query[:60]}...")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    docs = retriever.invoke(query)

    if not docs:
        logger.warning("No relevant documents found for this query.")
        return "No relevant information found in the knowledge base."

    # Merge retrieved page contents into one context block
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    logger.info(f"Retrieved {len(docs)} relevant chunks.")
    return context


# ─── 7. Full Ingestion Pipeline ───────────────────────────────────────────────
def run_ingestion_pipeline(file_path: str) -> dict:
    """
    Master function: PDF → Chunks → Embeddings → ChromaDB
    Called by main.py when the user clicks 'Process PDF'.
    Returns a status dict with message and chunk count.
    """
    try:
        documents  = load_pdf(file_path)
        chunks     = split_chunks(documents)
        embeddings = create_embeddings()
        store_chromadb(chunks, embeddings)

        return {
            "status": "success",
            "message": f"PDF processed successfully. {len(chunks)} chunks stored in ChromaDB.",
            "chunk_count": len(chunks),
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {"status": "error", "message": str(e)}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}
