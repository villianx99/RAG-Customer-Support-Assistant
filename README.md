# 🤖 RAG-Based Customer Support Assistant

**Powered by LangGraph · ChromaDB · Groq LLM · Human-in-the-Loop Escalation**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)

---

## 📌 Project Overview

An intelligent AI customer support assistant that answers user queries grounded in **uploaded PDF documents** using Retrieval-Augmented Generation (RAG). The system uses **LangGraph** for stateful workflow orchestration and includes **Human-in-the-Loop (HITL)** escalation — automatically routing sensitive or urgent queries to a human agent with a generated support ticket.

### Why This Project?

Traditional chatbots rely on scripted responses and fail on nuanced questions. Standard LLMs hallucinate because they lack domain knowledge. This system solves both problems by combining **document-grounded retrieval** with **agentic workflow routing**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 PDF Ingestion | Upload any PDF — text is extracted, chunked, and embedded |
| 🧠 RAG Pipeline | Semantic search retrieves relevant context before LLM generation |
| 🔀 LangGraph Workflow | Stateful graph with conditional routing (normal RAG vs. escalation) |
| 🧑‍💼 HITL Escalation | Urgent/complaint queries auto-escalate with ticket ID and priority |
| 💬 Streamlit UI | Clean, professional chat interface with real-time responses |
| 📊 Ticket Logging | All escalations saved to `logs/tickets.json` for audit |

---

## 🏗️ Architecture

```
User Query → Input Node → [Intent Router]
                              │
                    ┌─────────┴──────────┐
                 Normal             Escalate
                    │                    │
              Process Node         HITL Node
             (RAG + Groq LLM)    (Ticket Gen)
                    │                    │
                    └─────────┬──────────┘
                         Output Node
                              │
                         Final Response
```

---

## 📂 Project Structure

```
Final_RAG_Project/
│
├── data/                          # Uploaded PDFs + ChromaDB storage
│   └── vectordb/                  # Persistent vector store
│
├── docs/                          # Project documentation
│   ├── HLD.txt                    # High-Level Design
│   ├── LLD.txt                    # Low-Level Design
│   └── Technical_Documentation.txt
│
├── logs/                          # Execution logs + ticket records
│   ├── app.log                    # Auto-generated runtime log
│   └── tickets.json               # HITL escalation tickets
│
├── src/                           # Core application modules
│   ├── __init__.py
│   ├── main.py                    # Orchestrator (process_pdf, ask_query)
│   ├── workflow.py                # LangGraph state graph + nodes
│   ├── retriever.py               # PDF loading, chunking, ChromaDB ops
│   ├── hitl.py                    # Escalation ticket generation
│   └── utils.py                   # Logger, text cleaning, helpers
│
├── app.py                         # Streamlit frontend UI
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not committed to git)
└── README.md                      # This file
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.10+** installed
- **Groq API Key** — Free at [console.groq.com](https://console.groq.com)

### Step 1 — Clone / Download

```bash
git clone <your-repo-url>
cd Final_RAG_Project
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure API Key

Open the `.env` file and replace the placeholder:

```
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### Step 4 — Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🖥️ How to Use

1. **Upload a PDF** — Click the file uploader in the sidebar and select your document.
2. **Process the PDF** — Click the "⚡ Process PDF" button. Wait for the success message.
3. **Ask Questions** — Type your question in the text area and click "🚀 Submit".
4. **View Responses** — Normal queries get RAG answers (🟢 green badge). Urgent/complaint queries get escalated (🔴 red badge) with a ticket ID.

---

## 🧪 Sample Queries

| Query | Expected Behavior |
|---|---|
| "What is the refund policy?" | ✅ RAG answer from PDF |
| "Summarize the document" | ✅ RAG answer from PDF |
| "I need an urgent refund!" | 🔴 Escalated → Ticket generated |
| "Let me speak to a manager" | 🔴 Escalated → Ticket generated |
| "What are your business hours?" | ✅ RAG answer from PDF |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| LangChain | LLM orchestration, document loaders, vector store integrations |
| LangGraph | Stateful workflow graph with conditional edges |
| ChromaDB | Local vector database for semantic search |
| Groq (Llama 3.3 70B) | Ultra-fast LLM inference |
| HuggingFace (MiniLM) | Sentence embeddings for semantic similarity |
| Streamlit | Web UI framework |

---

## 🔮 Future Enhancements

- Multi-PDF support with source attribution
- Conversation memory across turns using LangGraph checkpointing
- Voice input/output integration (Whisper + TTS)
- Admin dashboard for ticket management
- Cloud deployment (Docker + AWS/GCP)
- WhatsApp/Slack bot integration
- Feedback loop for continuous improvement

---

## 📄 License

This project was built as a final internship/academic submission. Feel free to reference and build upon it.

---

## 👤 Author

**Arpit** — Final Year Internship Project  
Built with ❤️ using LangGraph, ChromaDB, and Groq
