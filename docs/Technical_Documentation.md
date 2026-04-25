==============================================================================
   TECHNICAL DOCUMENTATION
   RAG-Based Customer Support Assistant using LangGraph with HITL
==============================================================================

==============================================================================
A. INTRODUCTION TO RETRIEVAL-AUGMENTED GENERATION (RAG)
==============================================================================

Retrieval-Augmented Generation (RAG) is an architectural pattern introduced
by Facebook AI Research (Lewis et al., 2020) that enhances the capabilities
of Large Language Models by combining them with an external knowledge
retrieval system.

In a standard LLM interaction, the model relies solely on its parametric
memory — the knowledge baked into its weights during pre-training. This
memory is static, potentially outdated, and lacks private or domain-specific
information.

RAG addresses this by adding a retrieval step before generation:

  1. RETRIEVE: Given a user query, search an external knowledge base to find
     the most relevant documents or passages.
  2. AUGMENT: Inject the retrieved content into the LLM's prompt as context.
  3. GENERATE: The LLM produces an answer grounded in the retrieved context
     rather than relying on its own training data.

This pattern is particularly effective for enterprise applications where
accuracy, auditability, and freshness of information are critical.

==============================================================================
B. WHY RAG INSTEAD OF A NORMAL CHATBOT
==============================================================================

  Normal Chatbot / Vanilla LLM         RAG-Based System
  ────────────────────────────────────  ────────────────────────────────────
  Uses only pre-trained knowledge       Uses live, up-to-date documents
  Cannot access private/company data    Reads company PDFs, manuals, FAQs
  Prone to hallucination                Grounded in retrieved evidence
  Answers may be outdated               Answers reflect latest documents
  No source attribution                 Can cite specific document sections
  Expensive to update (fine-tuning)     Update by simply changing documents
  Generic responses                     Domain-specific, accurate responses

Key Insight: RAG gives an LLM an "open-book exam" instead of a
"closed-book exam." The quality of answers improves dramatically when the
model has the right reference material.

==============================================================================
C. HOW EMBEDDINGS WORK
==============================================================================

Embeddings are dense numerical representations of text in a high-dimensional
vector space. The core principle is that semantically similar text will have
similar embeddings (close together in vector space).

Process:
  1. A sentence-transformer model (e.g., all-MiniLM-L6-v2) takes raw text
     as input.
  2. The model processes the text through transformer layers and produces a
     fixed-size vector (384 dimensions for MiniLM).
  3. This vector captures the semantic meaning — not just keywords, but the
     underlying intent and topic of the text.

Example:
  "How do I return a product?"  →  [0.12, -0.45, 0.78, ..., 0.33]
  "What is the refund policy?"  →  [0.11, -0.43, 0.76, ..., 0.31]
  "What is the weather today?"  →  [-0.67, 0.22, -0.08, ..., 0.89]

The first two vectors would be very close (high cosine similarity) because
they are semantically related, even though they share no exact keywords.
The third vector would be far away.

In this project:
  - Model: all-MiniLM-L6-v2 (HuggingFace Sentence Transformers)
  - Dimensions: 384
  - Normalization: L2-normalized for cosine similarity
  - Execution: Runs locally on CPU — no API calls needed

==============================================================================
D. HOW ChromaDB WORKS
==============================================================================

ChromaDB is an open-source, AI-native vector database designed for storing
and querying embedding vectors. It is used in this project as the
persistent knowledge store.

How it works:
  1. STORE: When a PDF is processed, each text chunk is embedded and stored
     as a vector alongside its original text content and metadata.
  2. INDEX: ChromaDB builds an internal index (using HNSW — Hierarchical
     Navigable Small World graphs) to enable fast approximate nearest
     neighbor search.
  3. QUERY: When a user asks a question, the query is embedded using the
     same model, and ChromaDB finds the K closest stored vectors.
  4. RETURN: The original text chunks corresponding to those vectors are
     returned as search results.

Advantages of ChromaDB for this project:
  - Zero-configuration: No separate database server to install
  - Persistent storage: Survives application restarts (data/vectordb/)
  - LangChain integration: Native support via langchain-community
  - Lightweight: Suitable for single-machine deployment
  - Free and open-source: No API costs

==============================================================================
E. HOW RETRIEVAL IMPROVES ACCURACY
==============================================================================

Without retrieval, asking an LLM "What is your return policy?" forces the
model to guess based on general knowledge. The answer will be generic at
best, fabricated at worst.

With retrieval, the workflow becomes:

  1. User asks: "What is the return policy?"
  2. System embeds the question and searches ChromaDB
  3. ChromaDB returns: "Products may be returned within 30 days of purchase
     in original packaging. Electronics have a 15-day window..."
  4. The LLM receives this as context in its prompt:
     "Answer based ONLY on this context: [retrieved text]"
  5. The LLM produces a factual, grounded answer.

This eliminates hallucination because:
  - The model is explicitly constrained to the provided context
  - If no relevant context is found, it says "I don't know"
  - Every answer is traceable to a specific document passage

==============================================================================
F. HOW LANGGRAPH WORKFLOW WORKS
==============================================================================

LangGraph is an extension of LangChain that enables building stateful,
cyclical, multi-step AI workflows using graph architecture.

Key Concepts:
  - STATE: A typed dictionary (TypedDict) shared across all nodes. Each
    node reads and writes to this state.
  - NODES: Python functions that perform a specific action (e.g., classify
    intent, generate an answer, create a ticket).
  - EDGES: Connections between nodes. Can be direct (always follow) or
    conditional (choose based on state).
  - COMPILATION: The graph is compiled into an executable object that can
    be invoked with an initial state.

Our Graph Structure:

  ┌─────────────┐
  │ input_node  │  ← Receives query, detects intent
  └──────┬──────┘
         │
    [route_query]  ← Conditional edge
    ┌────┴─────┐
    │          │
    v          v
  ┌──────┐  ┌──────┐
  │ RAG  │  │ HITL │
  │ Node │  │ Node │
  └──┬───┘  └──┬───┘
     │          │
     └────┬─────┘
          v
  ┌─────────────┐
  │ output_node │  ← Logs and returns final state
  └─────────────┘
          │
         END

Why LangGraph over simple chains:
  - BRANCHING: Conditional edges allow dynamic routing based on state
  - STATE MANAGEMENT: TypedDict state is passed and updated across nodes
  - DETERMINISM: The graph structure is explicit and auditable
  - EXTENSIBILITY: Easy to add new nodes (e.g., sentiment analysis, caching)

==============================================================================
G. WHY HUMAN-IN-THE-LOOP (HITL) IS IMPORTANT
==============================================================================

No AI system is perfect. There are categories of user interactions where
automated responses are insufficient or potentially harmful:

  1. EMOTIONAL SITUATIONS: Angry or distressed customers need empathy that
     AI cannot authentically provide.
  2. FINANCIAL TRANSACTIONS: Refund processing, billing disputes, and fraud
     reports require authorization that AI should not have.
  3. LEGAL AND COMPLIANCE: Some queries touch regulatory areas where human
     judgment is legally required.
  4. COMPLEX EDGE CASES: Novel situations not covered by the knowledge base
     require creative human problem-solving.
  5. ESCALATION REQUESTS: When a customer explicitly asks for a human,
     honoring that request is a customer service principle.

Our HITL Implementation:
  - The router checks for escalation keywords (complaint, refund, fraud, etc.)
  - If triggered, the HITL node generates a support ticket with:
    - Unique ticket ID (TKT-YYYY-XXXXX)
    - Priority classification (CRITICAL / HIGH / MEDIUM)
    - Full chat context for the human agent
    - Timestamp and status tracking
  - The ticket is persisted to logs/tickets.json for the support team
  - The user receives an acknowledgment with their ticket ID

HITL Limitations:
  - Latency: Waiting for a human agent increases the total resolution time.
  - Cost: Human agents are significantly more expensive than automated LLM tokens.
  - Availability: Unlike the 24/7 AI, human agents may have restricted hours.
  - Scalability: Human intervention is the primary bottleneck in a high-volume system.
  - Subjectivity: Different agents may provide slightly different solutions to the same problem.

==============================================================================
H. PROMPT ENGINEERING USED
==============================================================================

Two primary prompts are engineered in this system:

PROMPT 1: RAG Answer Generation (process_node)
───────────────────────────────────────────────

  SYSTEM:
  "You are a professional and empathetic customer support AI assistant.
   Your job is to answer user questions ONLY using the provided context
   below. If the context does not contain enough information to answer
   accurately, say: 'I'm sorry, I don't have enough information about
   that in our knowledge base. Please contact our support team for
   further assistance.' Do NOT make up answers. Keep responses concise,
   clear, and helpful.

   CONTEXT:
   {context}"

  HUMAN:
  "{query}"

Design Decisions:
  - "ONLY using the provided context" — prevents hallucination
  - Explicit fallback instruction — ensures graceful failure
  - "Professional and empathetic" — sets appropriate tone
  - "Concise, clear, and helpful" — prevents verbose rambling

PROMPT 2: Intent Classification (input_node)
────────────────────────────────────────────
  Currently uses keyword matching for speed and determinism.
  In a future version, an LLM-based classifier could be added:

  "Analyze the following customer query. Classify it as 'normal' if it
   is a standard informational question. Classify it as 'escalate' if
   it contains complaints, urgency, requests for human agents, fraud
   mentions, or strong negative emotions. Output ONLY the word
   'normal' or 'escalate'."

==============================================================================
I. END-TO-END EXECUTION FLOW
==============================================================================

PHASE 1: PDF INGESTION (One-time per document)
───────────────────────────────────────────────
  1. User uploads PDF via Streamlit sidebar
  2. File saved to temp directory
  3. app.py calls main.process_pdf(temp_path)
  4. retriever.load_pdf() extracts text → List[Document]
  5. retriever.split_chunks() → 800-char chunks with 100-char overlap
  6. retriever.create_embeddings() → loads MiniLM model
  7. retriever.store_chromadb() → embeds and persists to data/vectordb/
  8. Success message returned to UI

PHASE 2: QUERY PROCESSING (Per user question)
──────────────────────────────────────────────
  1. User types question in chat area and clicks Submit
  2. app.py calls main.ask_query(query)
  3. retriever.retrieve_context() → embeds query, searches ChromaDB
  4. Top 4 relevant chunks concatenated into context string
  5. workflow.run_graph(query, context) invoked
  6. input_node: detects intent via keyword matching
  7. route_query: routes to process_node OR hitl_node

  PATH A (Normal Query):
    8a. process_node: Groq LLM generates answer using context
    9a. output_node: logs result
    10a. Response returned → displayed in green chat bubble

  PATH B (Escalation Query):
    8b. hitl_node: generates ticket, assigns priority
    9b. Ticket persisted to logs/tickets.json
    10b. output_node: logs result
    11b. Escalation message returned → displayed in red chat bubble

==============================================================================
J. PERFORMANCE EXPECTATIONS
==============================================================================

  Metric                           Expected Value
  ────────────────────────────────  ─────────────────────────────────────
  PDF ingestion (50-page doc)       15-30 seconds (includes embedding)
  First query (cold start)          3-5 seconds (model loading + search)
  Subsequent queries (warm)         1-3 seconds
  ChromaDB similarity search        < 200 milliseconds
  Groq LLM inference                500ms - 2 seconds
  Embedding model loading           2-4 seconds (first time)
  Embedding per chunk               < 50 milliseconds
  Memory usage (runtime)            ~500 MB (embedding model loaded)
  Disk usage (vector store)         ~10-50 MB per document

  Notes:
  - Performance tested on standard Windows machines with no GPU.
  - Groq inference is significantly faster than OpenAI or local models
    because Groq uses custom LPU (Language Processing Unit) hardware.
  - The embedding model (MiniLM) is compact at ~80 MB and runs
    efficiently on CPU.
  - ChromaDB uses HNSW indexing for sub-linear search complexity.

==============================================================================
K. SECURITY CONSIDERATIONS
==============================================================================

  1. API Key Management:
     - GROQ_API_KEY stored in .env file (not committed to version control)
     - python-dotenv loads keys at runtime only
     - .gitignore should include .env

  2. Data Privacy:
     - All data stays local (ChromaDB on disk, no cloud uploads)
     - utils.mask_sensitive() redacts PII from log files
     - Uploaded PDFs can be deleted after ingestion

  3. Input Validation:
     - Query length limited to 2000 characters
     - Empty queries rejected before processing
     - PDF format validated before loading

  4. Logging:
     - Rotating log files prevent disk exhaustion (2 MB cap)
     - Sensitive data masked before writing to logs

==============================================================================
L. DEPENDENCIES AND VERSION COMPATIBILITY
==============================================================================

  Package                  Min Version   Purpose
  ───────────────────────  ───────────   ────────────────────────────
  Python                   3.10          Runtime
  streamlit                1.32.0        Web UI framework
  langchain                0.3.0         LLM orchestration framework
  langchain-community      0.3.0         Community integrations
  langchain-core           0.3.0         Core abstractions
  langchain-groq           0.2.0         Groq LLM integration
  langchain-huggingface    0.1.0         HuggingFace embeddings
  langgraph                0.2.0         Stateful graph workflows
  chromadb                 0.5.0         Vector database
  pypdf                    4.0.0         PDF text extraction
  sentence-transformers    3.0.0         Embedding model framework
  python-dotenv            1.0.0         Environment variable loading

  All packages are installable via pip on Windows, macOS, and Linux.
  No system-level dependencies required beyond Python itself.

==============================================================================
M. CHALLENGES AND TRADE-OFFS
==============================================================================

  1. Retrieval Accuracy vs. Speed:
     - Increasing TOP_K improves context but increases LLM input tokens and 
       inference latency. We chose K=4 as a balance.
  
  2. Chunk Size vs. Context Quality:
     - 800-character chunks are large enough to contain semantic meaning but 
       small enough to fit multiple chunks into the LLM's context window 
       without truncation.
  
  3. Local Embedding vs. API Embedding:
     - We chose local MiniLM (CPU) for zero cost and privacy, despite slightly 
       lower semantic depth compared to OpenAI's larger `text-embedding-3-small`.

  4. Deterministic vs. Probabilistic Routing:
     - Keyword-based routing is 100% predictable and zero-latency but may 
       miss nuanced complaints that don't trigger specific words.

==============================================================================
N. TESTING STRATEGY
==============================================================================

  Our testing approach focuses on "Retrieval Grounding" and "Routing Accuracy."

  1. Ingestion Testing:
     - Verified extraction from single and multi-page PDFs.
     - Verified that overlapping chunks preserve meaning at boundaries.

  2. RAG Accuracy (Sample Queries):
     - Q: "What is the refund window?" -> Expect: "30 days" (verified)
     - Q: "How do I contact support?" -> Expect: Email/Phone from PDF (verified)
  
  3. Routing Testing (Sample Queries):
     - Q: "I am VERY angry" -> Expect: ESCALATED (verified)
     - Q: "I need a manager now" -> Expect: ESCALATED (verified)
     - Q: "What is your return policy?" -> Expect: RAG Answer (verified)

  4. Graceful Failure:
     - Q: "What is the weather in Paris?" (Not in PDF) -> Expect: "I don't have enough information..." (verified)

==============================================================================
O. FUTURE ENHANCEMENTS
==============================================================================

  1. Multi-Document Support: Allow users to upload multiple documents and 
     provide source attribution (e.g., "Answer found in 'Policy_v2.pdf'").

  2. Conversation Memory: Implement LangGraph Checkpointing to allow the bot 
     to remember previous turns in a conversation.

  3. LLM-Based Intent Classification: Replace keyword matching with an LLM 
     node for higher accuracy in detecting subtle complaints.

  4. Admin Dashboard: A separate Streamlit page for support agents to 
     view and close tickets from `tickets.json` in real-time.

  5. Cloud Deployment: Containerizing the app using Docker and deploying to 
     AWS/GCP for global availability.

==============================================================================
END OF TECHNICAL DOCUMENTATION
==============================================================================
