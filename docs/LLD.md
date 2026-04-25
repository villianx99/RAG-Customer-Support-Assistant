==============================================================================
   LOW LEVEL DESIGN (LLD)
   RAG-Based Customer Support Assistant using LangGraph with HITL
==============================================================================

==============================================================================
1. FOLDER STRUCTURE
==============================================================================

  Final_RAG_Project/
  │
  ├── data/
  │   ├── raw/                    # Uploaded PDF files (optional backup)
  │   └── vectordb/               # ChromaDB persistent storage
  │       ├── chroma.sqlite3      # Auto-generated index
  │       └── ...                 # Embedding data files
  │
  ├── docs/
  │   ├── HLD.txt                 # High Level Design
  │   ├── LLD.txt                 # Low Level Design (this file)
  │   └── Technical_Documentation.txt
  │
  ├── logs/
  │   ├── app.log                 # Runtime execution logs (rotated)
  │   └── tickets.json            # HITL escalation ticket records
  │
  ├── src/
  │   ├── __init__.py             # Package initializer
  │   ├── main.py                 # Application orchestrator
  │   ├── workflow.py             # LangGraph graph definition
  │   ├── retriever.py            # PDF + ChromaDB operations
  │   ├── hitl.py                 # Escalation / ticket module
  │   └── utils.py                # Shared helpers
  │
  ├── app.py                      # Streamlit frontend
  ├── requirements.txt            # Dependencies
  ├── .env                        # API keys
  └── README.md                   # Documentation

==============================================================================
2. MODULE WISE DESIGN
==============================================================================

2.1 app.py (Streamlit Frontend)
────────────────────────────────
  Responsibility:
    - Render the web UI (sidebar + main chat area)
    - Handle PDF file upload and forward to main.process_pdf()
    - Capture user queries and forward to main.ask_query()
    - Render chat history with color-coded intent badges
    - Manage Streamlit session state (chat_history, pdf_processed)

  External dependencies: streamlit, src.main

2.2 src/main.py (Orchestrator)
──────────────────────────────
  Responsibility:
    - Bridge between UI layer and backend logic
    - process_pdf(): calls retriever.run_ingestion_pipeline()
    - ask_query(): calls retriever.retrieve_context() then workflow.run_graph()
    - Caches embedding model and vector store across queries

  External dependencies: src.retriever, src.workflow, src.utils

2.3 src/retriever.py (Ingestion + Retrieval)
─────────────────────────────────────────────
  Responsibility:
    - load_pdf(): Extract text from PDF
    - split_chunks(): Break into overlapping chunks
    - create_embeddings(): Initialize HuggingFace embedding model
    - store_chromadb(): Embed and persist chunks
    - load_chromadb(): Load existing vector store from disk
    - retrieve_context(): Semantic search for top-K chunks
    - run_ingestion_pipeline(): Full end-to-end ingestion

  External dependencies: langchain, chromadb, sentence-transformers

2.4 src/workflow.py (LangGraph Engine)
──────────────────────────────────────
  Responsibility:
    - Define GraphState TypedDict
    - Implement graph nodes: input_node, process_node, hitl_node, output_node
    - Implement conditional router: route_query()
    - Build and compile the StateGraph
    - run_graph(): Execute the compiled graph for a single query

  External dependencies: langgraph, langchain-groq, src.hitl

2.5 src/hitl.py (Escalation Module)
────────────────────────────────────
  Responsibility:
    - Generate unique ticket IDs (TKT-YYYY-XXXXX)
    - Classify ticket priority (CRITICAL / HIGH / MEDIUM)
    - Format user-facing escalation message
    - Persist ticket to logs/tickets.json
    - get_all_tickets(): Utility for future admin dashboard

  External dependencies: None (standalone module)

2.6 src/utils.py (Helpers)
──────────────────────────
  Responsibility:
    - setup_logger(): Configure rotating file + console logger
    - clean_text(): Sanitize PDF-extracted text
    - timestamp(): Human-readable time strings
    - mask_sensitive(): Redact PII from log output
    - truncate(): Shorten long strings for log preview

  External dependencies: None (stdlib only)

==============================================================================
3. CLASS AND FUNCTION DESIGN
==============================================================================

3.1 retriever.py Functions
──────────────────────────

  Function              Input                    Output                 Notes
  ────────────────────  ───────────────────────  ─────────────────────  ──────────────────
  load_pdf(path)        str (file path)          List[Document]         Uses PyPDFLoader
  split_chunks(docs)    List[Document]           List[Document]         800 chars, 100 overlap
  create_embeddings()   None                     HuggingFaceEmbeddings  MiniLM-L6-v2
  store_chromadb(       List[Document],          Chroma                 Persists to disk
    chunks, embeddings) HuggingFaceEmbeddings
  load_chromadb(emb)    HuggingFaceEmbeddings    Chroma                 From existing dir
  retrieve_context(     str, Chroma              str (context)          Top 4 chunks
    query, vectordb)
  run_ingestion_        str (file path)          dict                   Master pipeline
    pipeline(path)

3.2 workflow.py Functions
─────────────────────────

  Function              Input                    Output                 Notes
  ────────────────────  ───────────────────────  ─────────────────────  ──────────────────
  get_llm()             None                     ChatGroq               Singleton pattern
  input_node(state)     GraphState               GraphState             Sets intent field
  process_node(state)   GraphState               GraphState             Calls Groq LLM
  hitl_node(state)      GraphState               GraphState             Calls escalate_ticket
  output_node(state)    GraphState               GraphState             Logging only
  route_query(state)    GraphState               str                    "process_node"|"hitl_node"
  build_graph()         None                     CompiledGraph          Full graph assembly
  run_graph(query, ctx) str, str                 dict (final state)     Public entry point

3.3 hitl.py Functions
─────────────────────

  Function              Input                    Output                 Notes
  ────────────────────  ───────────────────────  ─────────────────────  ──────────────────
  _generate_ticket_id() None                     str (TKT-YYYY-XXXXX)  Random serial
  _determine_priority() str (query)              str (priority level)   Keyword-based
  _persist_ticket()     dict (ticket payload)    None                   Appends to JSON file
  escalate_ticket()     str, list                dict                   Main public function
  get_all_tickets()     None                     List[dict]             Read ticket log

3.4 main.py Functions
─────────────────────

  Function              Input                    Output                 Notes
  ────────────────────  ───────────────────────  ─────────────────────  ──────────────────
  _get_vectordb()       None                     Chroma                 Lazy-load + cache
  process_pdf(path)     str                      dict (status+message)  Full ingestion
  ask_query(query)      str                      dict (response+intent) Full query pipeline

==============================================================================
4. DATA STRUCTURES
==============================================================================

4.1 GraphState (LangGraph State Dictionary)
───────────────────────────────────────────

  class GraphState(TypedDict):
      query           : str       # Original user question
      context         : str       # Retrieved chunks from ChromaDB
      intent          : str       # "normal" | "escalate"
      response        : str       # Final response text
      ticket_id       : str       # Populated only on escalation
      requires_human  : bool      # True when HITL path taken

4.2 Ticket Payload (HITL)
─────────────────────────

  {
      "ticket_id"    : "TKT-2024-84291",
      "priority"     : "HIGH",
      "user_query"   : "I need an urgent refund!",
      "chat_history" : [{"role": "user", "content": "..."}],
      "status"       : "OPEN",
      "assigned_to"  : "Unassigned",
      "timestamp"    : "2024-10-25T14:30:00",
      "notes"        : "Auto-escalated by AI..."
  }

4.3 LangChain Document Object
─────────────────────────────

  Document(
      page_content = "...text from PDF...",
      metadata     = {"source": "file.pdf", "page": 0}
  )

4.4 Ingestion Result
────────────────────

  {
      "status"      : "success" | "error",
      "message"     : "PDF processed successfully. 45 chunks stored.",
      "chunk_count" : 45
  }

4.5 Query Result
────────────────

  {
      "response"  : "Based on the documentation...",
      "intent"    : "normal" | "escalate",
      "ticket_id" : ""
  }

==============================================================================
5. LANGGRAPH WORKFLOW SPECIFICATION
==============================================================================

5.1 Nodes
─────────

  Node Name       Type         Purpose
  ──────────────  ──────────   ────────────────────────────────────────
  input_node      Entry        Receive query, detect intent via keywords
  process_node    Processing   Generate answer using Groq LLM + context
  hitl_node       Processing   Generate escalation ticket
  output_node     Terminal     Log result, pass state through

5.2 Edges
─────────

  From            To              Type          Condition
  ──────────────  ──────────────  ────────────  ──────────────────────
  START           input_node      Direct        Always (entry point)
  input_node      process_node    Conditional   intent == "normal"
  input_node      hitl_node       Conditional   intent == "escalate"
  process_node    output_node     Direct        Always
  hitl_node       output_node     Direct        Always
  output_node     END             Direct        Always (termination)

5.3 Escalation Keywords
───────────────────────

  complaint, refund, urgent, angry, fraud, manager,
  furious, stolen, legal, chargeback, speak to human,
  human agent, disappointed, unacceptable

5.4 Routing Decision Function (route_query)
───────────────────────────────────────────

  INPUT:  GraphState with intent field set by input_node
  OUTPUT: "process_node" (if intent is "normal")
          "hitl_node"    (if intent is "escalate")

5.5 Advanced Routing & Generation Criteria (Missing Context / Low Confidence)
─────────────────────────────────────────────────────────────────────────────

  While the initial intent router passes explicit escalations (keywords) directly 
  to the HITL node, the `process_node` implements secondary criteria for complex 
  edge cases via prompt engineering:

  - Answer generation: Proceeds normally if context chunks yield sufficient factual overlap.
  - Missing context: If the vector search returns irrelevant chunks (similarity < threshold), 
    the system prompt instructs the LLM to output: "I'm sorry, I don't have enough 
    information about that in our knowledge base."
  - Complex query: Multi-part or ambiguous queries are instructed to be answered 
    concisely and direct the user to a human if unresolved.
  - Low confidence: The LLM temperature is deliberately set to 0.3. If the logits 
    suggest low probability of correctness, the system falls back to the "I don't know" 
    string, avoiding hallucination.

==============================================================================
6. HITL INTEGRATION LIFECYCLE
==============================================================================

  6.1 When Escalation is Triggered
  - Triggered actively by the user (using keywords like "manager", "urgent")
  - Triggered passively by the system detecting severe negative sentiment ("fraud")

  6.2 What Happens After Escalation
  - The workflow routes to `hitl_node`.
  - A Ticket ID and priority are randomly/deterministically generated.
  - The payload (including full chat history) is appended to `logs/tickets.json`.
  - The user is notified on the UI with their unique Ticket ID.

  6.3 How Human Response is Integrated
  - This architecture acts as the automated Level-1 triage. 
  - Submitting to `tickets.json` acts as a Webhook/Database insert for a 
    downstream ticketing system (e.g., Zendesk, Jira Service Desk).
  - A human agent consumes the `tickets.json` via their dashboard.
  - The agent reads the attached `user_query` and `chat_history`.
  - The human agent then replies via Email or traditional CS portal using the 
    user's linked account information (out of scope for this bot's UI).

==============================================================================
7. ERROR HANDLING STRATEGY
==============================================================================

  Error Scenario              Module        Handling Approach
  ────────────────────────    ────────────  ────────────────────────────────
  PDF file not found          retriever     FileNotFoundError → user message
  PDF is empty / no text      retriever     ValueError → prompt re-upload
  No chunks created           retriever     ValueError → check PDF content
  ChromaDB not initialized    main          FileNotFoundError → "process PDF first"
  GROQ_API_KEY missing        workflow      EnvironmentError → clear message
  Groq API timeout/failure    workflow      try/except → fallback message
  Query too long (>2000 ch)   main          Input validation → truncation warning
  Empty query                 main          Input validation → prompt re-entry
  No relevant chunks found    retriever     Lenient prompt failure ("I don't know")
  Ticket log corrupted        hitl          try/except → reset empty list
  Logger file permission      utils         Graceful fallback to console only

==============================================================================
8. CONFIGURATION CONSTANTS
==============================================================================

  Constant                Module          Value
  ──────────────────────  ──────────────  ──────────────
  CHUNK_SIZE              retriever       800 characters
  CHUNK_OVERLAP           retriever       100 characters
  TOP_K_RESULTS           retriever       4
  EMBEDDING_MODEL         retriever       all-MiniLM-L6-v2
  LLM_MODEL               workflow        llama-3.3-70b-versatile
  LLM_TEMPERATURE         workflow        0.3
  LLM_MAX_TOKENS          workflow        1024
  LOG_MAX_BYTES           utils           2 MB
  LOG_BACKUP_COUNT        utils           3

==============================================================================
END OF LOW LEVEL DESIGN DOCUMENT
==============================================================================
