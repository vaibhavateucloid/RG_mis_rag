# This is the main entrypoint for the user and will be a streamlit app which will internally call the rag_main.py

import streamlit as st
from rag_main import RAGSystem
import time

# Set page configuration
st.set_page_config(page_title="LumenAI RG Chat",
                   page_icon="https://s3.amazonaws.com/lumenai.eucloid.com/assets/images/icons/logo.svg", layout="wide",
                   initial_sidebar_state="auto", menu_items=None)

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    return RAGSystem()

# Add logo and back button to sidebar
def add_logo_btn1():
    logo_url = "https://lumenai.eucloid.com/assets/images/logo.svg"
    back_button_url = "https://product.lumenai.eucloid.com/home"

    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: flex-start; align-items: center; padding-bottom: 20px;">
            <a href="{back_button_url}" target="_self">
                <img src="https://s3.amazonaws.com/lumenai.eucloid.com/assets/images/icons/back-btn.svg" alt="<-" width="20" height="20" style="margin-right: 10px;">
            </a>
            <div style="text-align: center;">
                <a href="https://product.lumenai.eucloid.com/login" target="_self">
                    <img src="{logo_url}" alt="Logo" width="225" height="fit-content">
                </a>
            </div>
        </div>
    """,
        unsafe_allow_html=True
    )

def main():
    add_logo_btn1()

    # --- Sidebar: Instructions & Inputs & Buttons ---
    instructions_md = """
    ### Instructions:
    1. This is a conversational chat bot with context of MIS ppt data.
    2. Please try to ask precise questions.
    """
    st.sidebar.markdown(instructions_md)

    # Initialize RAG system
    rag = initialize_rag()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main chat interface
    st.title("💬 LumenAI RG Chat")
    st.markdown("Ask me anything about your MIS data!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input("Ask your question about MIS data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare context for follow-up questions
                conversation_context = ""
                if len(st.session_state.messages) > 1:
                    # Get last few exchanges for context
                    recent_messages = st.session_state.messages[-10:]  # Last 5 exchanges
                    for msg in recent_messages:
                        conversation_context += f"{msg['role']}: {msg['content']}\n"
                
                # Create enhanced question with context for follow-ups
                if conversation_context:
                    enhanced_question = f"Previous conversation:\n{conversation_context}\nCurrent question: {question}"
                else:
                    enhanced_question = question
                
                # Get response from RAG system
                result = rag.query(enhanced_question)
                response = result['answer']
                
                # Display the response
                st.markdown(response)
                
                # Show sources if available (sorted by relevance in descending order)
                if result.get('sources'):
                    # Sort sources by relevance score in descending order (highest first)
                    sorted_sources = sorted(result['sources'], key=lambda x: x['score'], reverse=True)
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sorted_sources, 1):
                            st.markdown(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button in sidebar
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()