"""
app.py — Streamlit Frontend UI
RAG-Based Customer Support Assistant with LangGraph + HITL
Author: Final Year Internship Project
"""

import streamlit as st
import os
import sys
import tempfile

# ─── Add src/ to path so imports work cleanly ─────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main import process_pdf, ask_query

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
body { background-color: #0f1117; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #16193a 100%);
    border-right: 1px solid #2e3155;
}

/* ── Title Banner ── */
.title-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.title-banner h1 { color: white; font-size: 2rem; margin: 0; }
.title-banner p  { color: #d4d8ff; margin: 4px 0 0 0; font-size: 1rem; }

/* ── Chat Bubbles ── */
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px;
    margin: 12px 0; max-width: 80%; float: right; clear: both;
    box-shadow: 0 4px 12px rgba(102,126,234,0.3);
}
.chat-bot {
    background: #1e2130; color: #e4e6f0; padding: 14px 18px;
    border-radius: 18px 18px 18px 4px; margin: 12px 0; max-width: 80%;
    float: left; clear: both; border-left: 3px solid #667eea;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.chat-escalate {
    background: linear-gradient(135deg, #f97316, #dc2626);
    color: white; padding: 14px 18px; border-radius: 18px 18px 18px 4px;
    margin: 12px 0; max-width: 80%; float: left; clear: both;
    border-left: 3px solid #f97316;
    box-shadow: 0 4px 12px rgba(249,115,22,0.3);
}
.clearfix { clear: both; }

/* ── Status Badges ── */
.badge-normal    { background:#22c55e; color:white; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:700; }
.badge-escalate  { background:#ef4444; color:white; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:700; }

/* ── Info Cards ── */
.info-card {
    background: #1e2130; border: 1px solid #2e3155; border-radius: 12px;
    padding: 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Knowledge Base")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload the company policy, product manual, or FAQ document.",
    )

    if uploaded_file:
        st.markdown(f"**File:** `{uploaded_file.name}`")
        st.markdown(f"**Size:** `{round(uploaded_file.size / 1024, 1)} KB`")

    process_btn = st.button("⚡ Process PDF", use_container_width=True, type="primary")

    if process_btn:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            with st.spinner("Processing PDF... Building knowledge base..."):
                try:
                    # Save uploaded PDF to a temp file
                    suffix = ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    result = process_pdf(tmp_path)

                    if result["status"] == "success":
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")

    st.markdown("---")

    if st.session_state.pdf_processed:
        st.markdown(f"""
        <div class="info-card">
            <b>📄 Active Document</b><br>
            <small>{st.session_state.pdf_name}</small><br><br>
            <span class="badge-normal">READY</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            <b>📄 No Document Loaded</b><br>
            <small>Upload and process a PDF to begin.</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Routing Logic")
    st.markdown("""
    <div class="info-card">
        <b>🟢 Normal Route</b><br><small>Standard questions → RAG Answer</small><br><br>
        <b>🔴 Escalation Triggers</b><br>
        <small>complaint · refund · urgent · angry · fraud · manager</small>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ─── MAIN AREA ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-banner">
    <h1>🤖 AI Customer Support Assistant</h1>
    <p>Powered by RAG · LangGraph · Groq LLM · ChromaDB · Human-in-the-Loop Escalation</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="info-card">📚 <b>RAG Pipeline</b><br><small>Answers grounded in your documents</small></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="info-card">🔀 <b>Smart Routing</b><br><small>LangGraph conditional workflow</small></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="info-card">🧑‍💼 <b>HITL Escalation</b><br><small>Human takeover for complex cases</small></div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── CHAT HISTORY ─────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:48px; color:#4a5080;">
            <div style="font-size:3rem;">💬</div>
            <b>No conversation yet</b><br>
            <small>Upload a PDF and ask your first question below.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            intent = msg.get("intent", "normal")

            if role == "user":
                st.markdown(f'<div class="chat-user">🙋 {content}</div><div class="clearfix"></div>', unsafe_allow_html=True)
            elif role == "assistant":
                badge = f'<span class="badge-escalate">ESCALATED</span>' if intent == "escalate" else f'<span class="badge-normal">RAG</span>'
                bubble_class = "chat-escalate" if intent == "escalate" else "chat-bot"
                st.markdown(f'<div class="{bubble_class}">🤖 {badge}<br><br>{content}</div><div class="clearfix"></div>', unsafe_allow_html=True)

# ─── QUERY INPUT ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_area(
        "💬 Ask a Question",
        placeholder="E.g. What is the return policy? | I need an urgent refund...",
        height=90,
        help="Ask anything related to the uploaded document.",
    )
    submit_btn = st.form_submit_button("🚀 Submit", use_container_width=True, type="primary")

if submit_btn:
    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload and process a PDF document first.")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("🧠 Processing your query through LangGraph workflow..."):
            try:
                result = ask_query(user_query)
                response_text = result.get("response", "No response generated.")
                intent = result.get("intent", "normal")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "intent": intent,
                })
                st.rerun()

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5080; font-size:0.85rem; padding: 8px;">
    RAG Customer Support Assistant · Built with LangGraph + ChromaDB + Groq · Final Internship Project
</div>
""", unsafe_allow_html=True)
