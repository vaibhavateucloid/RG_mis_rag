import streamlit as st
from rag_main import RobustRAGSystem, run_data_processing, Config
import os
import hashlib

st.set_page_config(page_title="Robust RAG Chat", layout="wide")

def load_documents_from_folder(folder_path="documents"):
    documents = []
    print(f"[APP] Loading documents from folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            try:
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [{"content": page.extract_text() or ""} for page in reader.pages]
                documents.append({"filename": filename, "pages": pages})
                print(f"[APP] Loaded PDF: {filename} ({len(pages)} pages)")
            except Exception as e:
                print(f"[APP] Failed to load {filename}: {e}")
                st.warning(f"Failed to load {filename}: {e}")
        elif filename.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"filename": filename, "pages": [{"content": content}]})
                print(f"[APP] Loaded TXT: {filename}")
            except Exception as e:
                print(f"[APP] Failed to load {filename}: {e}")
                st.warning(f"Failed to load {filename}: {e}")
    print(f"[APP] Total documents loaded: {len(documents)}")
    return documents

def get_documents_hash(documents):
    # Create a hash of all filenames and their sizes for change detection
    hash_md5 = hashlib.md5()
    for doc in sorted(documents, key=lambda d: d['filename']):
        hash_md5.update(doc['filename'].encode())
    return hash_md5.hexdigest()

@st.cache_resource
def get_rag_system():
    # Load documents and compute hash for change detection
    folder_path = "documents"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    documents = load_documents_from_folder(folder_path)
    documents_hash = get_documents_hash(documents) if documents else None
    vector_store_exists = os.path.exists(Config.vector_store_path) and os.listdir(Config.vector_store_path)
    print(f"[APP] Vector store exists: {vector_store_exists}")
    if not vector_store_exists or documents_hash is not None:
        if not documents:
            print("[APP] No Documents found in the 'documents' folder.")
            st.error("No Documents found in the 'documents' folder.")
            st.stop()
        print("[APP] Running data processing pipeline...")
        run_data_processing(documents)
        print("[APP] Data processing complete.")
    else:
        print("[APP] Using existing vector store.")
    return RobustRAGSystem()

rag_system = get_rag_system()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 Robust RAG Chat with Context & Reasoning")

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("reasoning_steps"):
            with st.expander("🤔 See my thinking"):
                for step in entry["reasoning_steps"]:
                    st.markdown(f"- {step}")

# User input
if prompt := st.chat_input("Ask a question about your financial documents..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Build context from previous messages (if needed)
    # For now, just concatenate previous Q&A
    context = "\n".join(
        f"User: {e['content']}\nAssistant: {e.get('answer', '')}"
        for e in st.session_state.chat_history if e["role"] == "assistant"
    )

    # Query the RAG system
    with st.spinner("Thinking..."):
        result = rag_system.query(prompt, k=15)
        answer = result["answer"]
        reasoning_steps = result.get("reasoning_steps", [])

    # Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "reasoning_steps": reasoning_steps,
        "answer": answer
    })

    # Rerun to display the new message
    st.rerun() 