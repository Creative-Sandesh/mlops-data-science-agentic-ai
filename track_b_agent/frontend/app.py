import streamlit as st
import requests

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title(" Production AI Assistant")

# Backend API configuration
BACKEND_URL = "http://backend:8000"

# Sidebar - Document Ingestion
with st.sidebar:
    st.header("📄 Knowledge Base Ingestion")
    doc_text = st.text_area("Paste document content to index into RAG:")
    if st.button("Ingest Document"):
        if doc_text:
            res = requests.post(f"{BACKEND_URL}/ingest", json={"document_text": doc_text})
            if res.status_code == 200:
                st.success(f"Indexed! Chunks: {res.json().get('chunks_processed')}")
            else:
                st.error("Ingestion failed.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"user_query": prompt, "use_rag": True}
            )
            if response.status_code == 200:
                data = response.json()
                st.markdown(data["response"])
                
                # Show performance/reliability metrics
                st.caption(
                    f"⚡ Cached: {data.get('cached', False)} | "
                    f"🛠️ Provider: {data.get('provider_used', 'N/A')}"
                )
                st.session_state.messages.append({"role": "assistant", "content": data["response"]})
            else:
                st.error("Error communicating with AI Backend API.")