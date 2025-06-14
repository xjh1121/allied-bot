
import streamlit as st
import openai
import faiss
import json
import numpy as np

# Load FAISS index and metadata once
@st.cache_resource
def load_faiss_and_metadata():
    index = faiss.read_index("faiss_index.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_faiss_and_metadata()

# Sidebar for API Key entry (not hardcoded for security)
st.sidebar.title("API Settings")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.title("🏠 Allied Title AI Assistant")
st.markdown("Ask any question about title insurance, closing, policies, or processes.")

# User input
question = st.text_input("Your question", "")

def get_response(question, top_k=3):
    embed = openai.Embedding.create(
        input=question,
        model="text-embedding-3-small",
        api_key=openai_api_key
    )["data"][0]["embedding"]
    D, I = index.search(np.array([embed], dtype=np.float32), top_k)
    context = "\n\n".join([metadata[str(i)] for i in I[0] if str(i) in metadata])
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant for a real estate title company."},
                  {"role": "user", "content": prompt}],
        api_key=openai_api_key
    )
    return response.choices[0].message["content"]

# Handle user question
if question:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = get_response(question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
