"""
app.py — Phase 8: Streamlit UI
--------------------------------
A simple web interface for the Traffic RAG pipeline.
Run with: streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from retriever import TrafficRetriever
from generator import generate_answer

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Guntur Traffic RAG",
    page_icon   = "🚦",
    layout      = "wide",
)

# ── Styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .answer-box {
        background: #1e2130;
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #e0e0e0;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .strategy-badge {
        display: inline-block;
        background: #2a3a5c;
        color: #7eb8f7;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .location-badge {
        display: inline-block;
        background: #1e3a2a;
        color: #6fcf97;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .chunk-box {
        background: #161922;
        border: 1px solid #2a2d3e;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        color: #adb5bd;
        font-size: 0.88rem;
        line-height: 1.6;
        font-family: monospace;
    }
    .meta-label {
        color: #888;
        font-size: 0.82rem;
        margin-bottom: 4px;
    }
    .example-q {
        cursor: pointer;
        color: #7eb8f7;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache the retriever so it only loads once ────────────────────
@st.cache_resource
def load_retriever():
    base      = os.path.dirname(__file__)
    processed = os.path.join(base, 'data', 'processed')
    vectordb  = os.path.join(base, 'vectorstore')
    return TrafficRetriever(processed, vectordb)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Traffic RAG")
    st.markdown("Query traffic survey data from **84 locations** on state highways in Guntur District, Andhra Pradesh.")
    st.divider()

    st.markdown("**Dataset**")
    st.markdown("- Source: AIKosh / IndiaAI\n- 84 XLS survey files\n- Vehicle counts by type & direction")

    st.divider()
    st.markdown("**Retrieval Strategies**")
    st.markdown("""
- 🏆 **Superlative** — highest / lowest across all 84 locations
- 📍 **Location Specific** — by P-code or road name
- ⚖️ **Comparison** — two locations side by side
- 🔍 **Semantic** — vector search for everything else
    """)

    st.divider()
    st.markdown("**RAGAS Evaluation**")
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", "0.941")
    col2.metric("Answer Relevancy", "0.893")

    st.divider()
    st.caption("Built with Python · ChromaDB · Groq · RAGAS")


# ── Main area ────────────────────────────────────────────────────
st.markdown("# Guntur District Traffic Query System")
st.markdown("Ask a natural language question about highway traffic counts across Guntur District.")

# Example queries
st.markdown("**Try an example:**")
examples = [
    "Which road has the highest truck traffic?",
    "What is the peak hour at Nadikudi?",
    "Compare P526 and P538",
    "Which roads were surveyed near Tadikonda?",
    "Which location has the lowest AADT?",
    "Tell me about location P606",
]

cols = st.columns(3)
for i, example in enumerate(examples):
    if cols[i % 3].button(example, use_container_width=True):
        st.session_state['query'] = example

st.divider()

# Query input
query = st.text_input(
    label       = "Your question",
    value       = st.session_state.get('query', ''),
    placeholder = "e.g. Which road has the highest truck traffic?",
    label_visibility = "collapsed",
)

ask = st.button("Ask", type="primary", use_container_width=False)

# ── Run pipeline ─────────────────────────────────────────────────
if ask and query.strip():
    with st.spinner("Retrieving and generating answer..."):
        retriever = load_retriever()
        retrieved = retriever.retrieve(query)
        response  = generate_answer(query, retrieved)

    answer    = response['answer']
    locations = response['locations']
    strategy  = response['strategy']

    # Strategy + location badges
    st.markdown("---")
    badge_html = ""
    for s in strategy:
        badge_html += f'<span class="strategy-badge">⚙ {s}</span>'
    for loc in locations:
        badge_html += f'<span class="location-badge">📍 {loc}</span>'
    st.markdown(f'<div class="meta-label">Retrieval info</div>{badge_html}', unsafe_allow_html=True)

    # Answer
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Retrieved chunks in expander
    with st.expander(f"View retrieved context ({len(retrieved)} chunk{'s' if len(retrieved) != 1 else ''})"):
        for i, chunk in enumerate(retrieved, 1):
            st.markdown(
                f'<div class="chunk-box"><strong>Chunk {i}</strong> — {chunk.chunk_id}<br><br>{chunk.text[:600]}{"..." if len(chunk.text) > 600 else ""}</div>',
                unsafe_allow_html=True,
            )

elif ask and not query.strip():
    st.warning("Please enter a question first.")