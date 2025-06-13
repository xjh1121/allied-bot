import streamlit as st
import openai
import faiss
import json
import numpy as np

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.index")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Set API key securely from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_response(question, top_k=3):
    embed = openai.Embedding.create(
        input=question,
        model="text-embedding-3-small"
    )
    vector = np.array(embed["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)
    _, indices = index.search(vector, top_k)
    chunks = [metadata[i]["content"] for i in indices[0]]

    system = """You are a helpful real estate assistant. Only answer using the provided company documents. No speculation. No emojis."""
    user = f"""Documents:\n{chr(10).join(chunks)}\n\nQuestion: {question}"""

    chat = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return chat["choices"][0]["message"]["content"]

# Streamlit UI
st.set_page_config(page_title="Allied Bot", layout="centered")
st.title("🤖 Allied Bot")

user_input = st.text_input("Ask a question about title, closings, or the company:")
if user_input:
    with st.spinner("Thinking..."):
        answer = get_response(user_input)
        st.markdown("### Answer")
        st.write(answer)