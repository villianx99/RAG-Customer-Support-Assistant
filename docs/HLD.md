==============================================================================
   HIGH LEVEL DESIGN (HLD)
   RAG-Based Customer Support Assistant using LangGraph with HITL
==============================================================================

==============================================================================
1. SYSTEM OVERVIEW
==============================================================================

1.1 Problem Statement
---------------------
Customer support teams face a growing volume of repetitive queries that
waste human agent time and increase response latency. Generic chatbots use
scripted responses that break on nuanced questions. Standalone LLMs
hallucinate because they lack access to company-specific policies and data.

There is a need for an intelligent support system that:
  - Answers questions accurately using company documents
  - Distinguishes between routine queries and urgent escalations
  - Routes critical issues to human agents without dropping context

1.2 Scope
---------
This project delivers:
  - A PDF-based knowledge ingestion pipeline
  - A vector search engine for semantic retrieval
  - An LLM-powered answer generation layer
  - A stateful LangGraph workflow with conditional routing
  - A Human-in-the-Loop (HITL) escalation module
  - A web-based chat interface via Streamlit

Out of scope for this version:
  - Multi-tenant / multi-user authentication
  - Real-time human agent dashboard (tickets logged to JSON)
  - Voice or multi-modal input

1.3 Objectives
--------------
  O1. Build a RAG pipeline that grounds LLM responses in uploaded documents.
  O2. Use LangGraph to orchestrate a deterministic, auditable workflow.
  O3. Implement conditional routing that escalates sensitive queries.
  O4. Provide a clean, professional UI for demo and evaluation.
  O5. Ensure the system is portable and runs on Windows with minimal setup.

==============================================================================
2. ARCHITECTURE DIAGRAM
==============================================================================

  +------------------------------------------------------------------+
  |                      STREAMLIT WEB UI                             |
  |  ┌──────────────┐                    ┌─────────────────────┐     |
  |  │ PDF Upload   │                    │  Chat Interface      │     |
  |  │ + Process Btn│                    │  (Query + Response)  │     |
  |  └──────┬───────┘                    └──────────┬──────────┘     |
  +---------|---------------------------------------|----------------+
            │ (file bytes)                          │ (user query)
            v                                       v
  ┌──────────────────┐                   ┌──────────────────────┐
  │  INGESTION       │                   │   QUERY PIPELINE      │
  │  PIPELINE        │                   │                       │
  │                  │                   │  1. Embed query       │
  │  1. PyPDFLoader  │                   │  2. ChromaDB search   │
  │  2. Chunker      │                   │  3. Retrieve top-K    │
  │  3. Embeddings   │                   │     context chunks    │
  │  4. ChromaDB     │                   │                       │
  │     store        │                   └───────────┬───────────┘
  └──────────┬───────┘                               │
             │                                       │ (context string)
             v                                       v
  ┌──────────────────┐               ┌──────────────────────────────┐
  │  ChromaDB        │               │   LANGGRAPH WORKFLOW ENGINE   │
  │  Vector Store    │               │                               │
  │  (persistent     │◄──────────────│  ┌────────────┐               │
  │   local disk)    │               │  │ INPUT NODE │               │
  └──────────────────┘               │  └─────┬──────┘               │
                                     │        │ intent detection     │
                                     │   ┌────┴─────┐               │
                                     │   │ ROUTER   │               │
                                     │   └──┬────┬──┘               │
                                     │ normal│    │escalate          │
                                     │      v    v                  │
                                     │  ┌─────┐ ┌──────┐            │
                                     │  │ RAG │ │ HITL │            │
                                     │  │NODE │ │ NODE │            │
                                     │  └──┬──┘ └──┬───┘            │
                                     │     └───┬───┘                │
                                     │         v                    │
                                     │   ┌───────────┐              │
                                     │   │OUTPUT NODE│              │
                                     │   └───────────┘              │
                                     └──────────────────────────────┘

==============================================================================
3. COMPONENT DESCRIPTIONS
==============================================================================

3.1 User Interface (Streamlit)
  - File uploader widget for PDF documents
  - Process button triggers ingestion pipeline
  - Chat-style text area for queries
  - Color-coded response bubbles (green = RAG, red = escalation)
  - Sidebar shows active document and routing logic summary

3.2 Document Loader (PyPDFLoader)
  - Accepts any standard PDF file
  - Extracts text page by page into LangChain Document objects
  - Handles multi-page documents and metadata extraction

3.3 Chunking Engine (RecursiveCharacterTextSplitter)
  - Splits documents into 800-character chunks with 100-character overlap
  - Uses hierarchical separators: paragraph → sentence → word
  - Overlap ensures no semantic context is lost at boundaries

3.4 Embedding Model (HuggingFace all-MiniLM-L6-v2)
  - Lightweight sentence-transformer model (80MB)
  - Produces 384-dimensional normalized vectors
  - Runs entirely on CPU — no GPU required
  - Downloaded once, cached locally for subsequent runs

3.5 Vector Database (ChromaDB)
  - Open-source, file-based vector database
  - Persistent storage to data/vectordb/ — survives restarts
  - Uses cosine similarity for nearest-neighbor search
  - Returns top-K most relevant chunks per query

3.6 Retriever
  - Converts user query to embedding vector
  - Performs similarity search against stored document vectors
  - Returns top 4 chunks as concatenated context string
  - Passes context to the LangGraph workflow

3.7 LangGraph Workflow Engine
  - StateGraph with typed state dictionary (GraphState)
  - 4 nodes: input_node, process_node, hitl_node, output_node
  - Conditional edge after input_node for intent-based routing
  - Compiled graph runs as a single synchronous invocation

3.8 LLM Layer (Groq — Llama 3.3 70B)
  - Groq provides ultra-low-latency inference via LPU hardware
  - Llama 3.3 70B Versatile model selected for quality + speed
  - Temperature 0.3 for focused, factual responses
  - System prompt constrains answers to provided context only

3.9 Conditional Router
  - Keyword-based fast path (zero-latency check)
  - Trigger words: complaint, refund, urgent, angry, fraud, manager, etc.
  - Returns "normal" or "escalate" to the LangGraph edge function

3.10 HITL Escalation Module
  - Generates unique ticket ID (format: TKT-YYYY-XXXXX)
  - Assigns priority level (CRITICAL / HIGH / MEDIUM)
  - Persists ticket payload to logs/tickets.json
  - Returns formatted escalation message to the user

==============================================================================
4. END-TO-END DATA FLOW
==============================================================================

INGESTION FLOW:
  PDF file → PyPDFLoader → List[Document] → RecursiveCharacterTextSplitter
  → List[Chunk] → HuggingFace Embeddings → ChromaDB (persist to disk)

QUERY FLOW:
  User query → Embed query → ChromaDB similarity search → Top-K chunks
  → Concatenate context → Pass to LangGraph

LANGGRAPH FLOW:
  GraphState(query, context) → input_node (intent detection)
  → route_query() → [process_node (Groq LLM) | hitl_node (ticket)]
  → output_node → Return response to UI

==============================================================================
5. TECHNOLOGY CHOICES & JUSTIFICATION
==============================================================================

  Technology              Justification
  ──────────────────────  ──────────────────────────────────────────────────
  Python 3.10+            Industry standard for AI/ML. Richest ecosystem.
  LangChain               Unified API for LLMs, loaders, vector stores.
  LangGraph               Cyclic stateful graphs — superior to linear chains.
  ChromaDB                Zero-config local vector DB. No server needed.
  Groq (Llama 3.3)        10x faster than OpenAI. Free tier available.
  HuggingFace MiniLM      Small, fast, accurate embeddings. No API needed.
  Streamlit               Rapid UI prototyping without frontend expertise.
  python-dotenv           Secure API key management via .env files.

==============================================================================
6. SCALABILITY CONSIDERATIONS
==============================================================================

  Concern                 Current Approach          Future Scale Strategy
  ──────────────────────  ────────────────────────  ──────────────────────────
  Large PDFs (500+ pg)    Batch chunking            Async ingestion workers
  Multiple users          Streamlit single-session  FastAPI + WebSocket backend
  Vector DB performance   ChromaDB local            Pinecone / Qdrant cloud
  LLM latency             Groq LPU (~1s)            Response caching / Redis
  Deployment              Local Windows             Docker + AWS ECS / K8s

==============================================================================
END OF HIGH LEVEL DESIGN DOCUMENT
==============================================================================
