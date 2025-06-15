import streamlit as st
import numpy as np
import faiss
import json
import os
import openai

# ── CONFIG ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"   # swap to gpt-4o or gpt-4 if desired
TOP_K           = 7               # identical to terminal bot
TEMPERATURE     = 0.0             # deterministic
# ────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a helpful, professional real estate assistant. Your job is to explain
topics related to title insurance and closings in a clear, accurate, and
easy-to-read way based only on the provided documents.

Formatting & Style Guidelines:
- Use bold section headers followed by clear bullet points or concise paragraphs.
- Keep answers professional, factual, and mobile-friendly.
- Do not include emojis, icons, tables, or suggest one product is better unless
  the document says so directly.
- If the answer isn't in the documents, say so and recommend the user contact
  Allied Title.
""".strip()

# ── LOAD INDEX + METADATA (cached) ─────────────────────────────────────
@st.cache_resource
def load_resources():
    index = faiss.read_index("faiss_index.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    metadata = [item.get("content", "") for item in meta_raw]
    return index, metadata

index, metadata = load_resources()

# ── SIDEBAR: API KEY ───────────────────────────────────────────────────
st.sidebar.header("🔑 API Settings")
user_key = st.sidebar.text_input("OpenAI API Key", type="password")
OPENAI_API_KEY = user_key or os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── UI ─────────────────────────────────────────────────────────────────
st.title("🏠 Allied Title AI Assistant")
st.markdown(
    "Ask any question about **title insurance, closings, fees, or Allied Title "
    "processes.** Answers come only from official Allied documents."
)
question = st.text_input("Your question")

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────
def semantic_search(query: str, k: int = TOP_K) -> str:
    """Return concatenated top-k document chunks for the prompt."""
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array(emb.data[0].embedding, dtype=np.float32)
    D, I = index.search(vec.reshape(1, -1), k)
    return "\n\n".join(metadata[i] for i in I[0] if i < len(metadata))

def answer_question(q: str) -> str:
    docs = semantic_search(q)
    user_prompt = f"Documents:\n{docs}\n\nQuestion:\n{q}\n"
    chat = client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = TEMPERATURE,
    )
    return chat.choices[0].message.content.strip()

# ── RUN ────────────────────────────────────────────────────────────────
if question:
    if not OPENAI_API_KEY:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                st.write(answer_question(question))
            except Exception as e:
                st.error(f"Error: {e}")
