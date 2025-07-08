# Enhanced Streamlit App with Live Thinking Display

import streamlit as st
from rag_main import RAGSystem  # Using the enhanced RAG system
import time
import logging

# Set page configuration
st.set_page_config(page_title="LumenAI RG Chat",
                   page_icon="https://s3.amazonaws.com/lumenai.eucloid.com/assets/images/icons/logo.svg", 
                   layout="wide",
                   initial_sidebar_state="auto", 
                   menu_items=None)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

def display_thinking_step(step_content):
    """Display a single thinking step with appropriate styling."""
    if "🧠 **THINKING**:" in step_content:
        step_text = step_content.replace("🧠 **THINKING**: ", "").strip()
        st.markdown(f"🧠 **Starting analysis...** {step_text}")
    elif "🔍 **THINKING**:" in step_content:
        step_text = step_content.replace("🔍 **THINKING**: ", "").strip()
        st.markdown(f"🔍 **Searching...** {step_text}")
    elif "📋 **THINKING**:" in step_content:
        step_text = step_content.replace("📋 **THINKING**: ", "").strip()
        if "sub-queries:" in step_text:
            # Handle sub-queries display specially
            parts = step_text.split("sub-queries:")
            if len(parts) > 1:
                st.markdown(f"📋 **Planning...** {parts[0].strip()} sub-queries:")
                # Display sub-queries in a nice format
                sub_queries_part = parts[1].strip()
                if sub_queries_part:
                    st.markdown(sub_queries_part)
        else:
            st.markdown(f"📋 **Planning...** {step_text}")
    elif "✅ **THINKING**:" in step_content:
        step_text = step_content.replace("✅ **THINKING**: ", "").strip()
        st.markdown(f"✅ **Progress...** {step_text}")
    elif "📊 **THINKING**:" in step_content:
        step_text = step_content.replace("📊 **THINKING**: ", "").strip()
        st.markdown(f"📊 **Analyzing...** {step_text}")
    elif "🤖 **THINKING**:" in step_content:
        step_text = step_content.replace("🤖 **THINKING**: ", "").strip()
        st.markdown(f"🤖 **Generating...** {step_text}")
    elif "⚠️ **THINKING**:" in step_content:
        step_text = step_content.replace("⚠️ **THINKING**: ", "").strip()
        st.warning(f"⚠️ {step_text}")
    elif "❌ **THINKING**:" in step_content:
        step_text = step_content.replace("❌ **THINKING**: ", "").strip()
        st.error(f"❌ {step_text}")
    else:
        # Fallback for any other format
        st.markdown(step_content)

def main():
    add_logo_btn1()
    logging.info("Application started.")

    # --- Sidebar: Instructions & Inputs & Buttons ---
    instructions_md = """
    ### Instructions:
    1. Ask direct questions for quick facts or analytical questions for deep insights.
    2. April 2024 - March 2025 data available.
    """
    st.sidebar.markdown(instructions_md)

    # Initialize RAG system
    rag = initialize_rag()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main chat interface
    st.title("💬 LumenAI RG Chat")
    st.markdown("Ask me anything about RateGain's financial performance - I'm your CFA!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "thinking_steps" in message:
                # Display thinking steps if they exist
                with st.expander("🧠 Thinking Process", expanded=False):
                    for step in message["thinking_steps"]:
                        display_thinking_step(step)
            
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")

    # Accept user input
    if question := st.chat_input("Ask about RateGain's financial performance..."):
        logging.info(f"User submitted question: {question}")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare context for follow-up questions
            conversation_context = ""
            if len(st.session_state.messages) > 1:
                # Get last few exchanges for context
                recent_messages = st.session_state.messages[-10:]  # Last 5 exchanges
                for msg in recent_messages:
                    if msg["role"] in ["user", "assistant"]:
                        conversation_context += f"{msg['role']}: {msg['content']}\n"
            
            # Create enhanced question with context for follow-ups
            if conversation_context:
                enhanced_question = f"Previous conversation:\n{conversation_context}\nCurrent question: {question}"
            else:
                enhanced_question = question
            
            # Initialize containers for live updates
            thinking_container = st.container()
            answer_container = st.empty()
            sources_container = st.empty()
            
            thinking_steps = []
            final_answer = ""
            final_sources = []
            
            # Create a placeholder for thinking steps
            thinking_placeholder = thinking_container.empty()
            spinner_placeholder = None
            # Process query with live thinking display
            for response in rag.query(enhanced_question, conversation_context):
                if response["type"] == "thinking":
                    thinking_steps.append(response["content"])
                    # Update thinking display in real-time with collapsible container
                    with thinking_placeholder.container():
                        with st.expander("🧠 **Live Thinking Process**", expanded=True):
                            subquery_end_idx = None
                            for idx, step in enumerate(thinking_steps):
                                display_thinking_step(step)
                                # Detect the last sub-query bullet point
                                if step.startswith("• "):
                                    subquery_end_idx = idx
                            # Show spinner after the last sub-query bullet point
                            if subquery_end_idx is not None and len(thinking_steps) == subquery_end_idx + 1:
                                spinner_placeholder = st.empty()
                                with spinner_placeholder.container():
                                    st.info("⏳ Analyzing and generating answer...")
                            else:
                                if spinner_placeholder:
                                    spinner_placeholder.empty()
                
                elif response["type"] == "answer":
                    final_answer = response["content"]
                    final_sources = response.get("sources", [])
                    
                    # Collapse the thinking process and show final answer
                    with thinking_placeholder.container():
                        with st.expander("🧠 **Thinking Process**", expanded=False):
                            for step in thinking_steps:
                                display_thinking_step(step)
                    
                    if spinner_placeholder:
                        spinner_placeholder.empty()
                    
                    answer_container.markdown(final_answer)
                    
                    # Show sources
                    if final_sources:
                        with sources_container.expander("📚 Sources"):
                            for i, source in enumerate(final_sources, 1):
                                st.markdown(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")
                
                elif response["type"] == "error":
                    # Clear thinking and show error
                    if spinner_placeholder:
                        spinner_placeholder.empty()
                    thinking_placeholder.empty()
                    answer_container.error(response["content"])
                    final_answer = response["content"]
                    logging.error(f"Error response from RAG system: {response['content']}")
            
            # Add assistant response to chat history with thinking steps
            assistant_message = {
                "role": "assistant", 
                "content": final_answer,
                "thinking_steps": thinking_steps,
                "sources": final_sources
            }
            st.session_state.messages.append(assistant_message)
            logging.info(f"Assistant response generated: {final_answer}")

    # Clear chat button in sidebar
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # System status in sidebar
    st.sidebar.markdown("---")
    if rag.is_ready():
        st.sidebar.success("✅ CFA System Ready")
    else:
        st.sidebar.error("❌ System Not Ready")
    logging.info("Application finished.")

if __name__ == "__main__":
    main()