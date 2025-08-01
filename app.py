# Enhanced Streamlit App with Live Thinking Display (Chainlit-parity version)

import streamlit as st
import logging
import asyncio
import time
from rag_main import EnhancedRAGSystem as RAGSystem


# --- Constants ---
MAX_MESSAGE_LENGTH = 50000
MAX_THINKING_LENGTH = 30000

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Q&A Cache ---
if "last_qa_cache" not in st.session_state:
    st.session_state.last_qa_cache = {"question": None, "answer": None, "sources": None}

def normalize_question(q):
    return q.strip().lower() if isinstance(q, str) else q

# --- RAG System ---
@st.cache_resource
def initialize_rag():
    return RAGSystem()

# --- Sidebar UI ---
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

# --- Thinking Step Formatting ---
def format_thinking_content(step_content):
    if "🧠 **THINKING**:" in step_content:
        return f"🧠 **Analysis**: {step_content.replace('🧠 **THINKING**: ', '').strip()}"
    elif "🔍 **THINKING**:" in step_content:
        content = step_content.replace("🔍 **THINKING**: ", "").strip()
        if "sub-queries" in content.lower():
            parts = content.split("sub-queries:")
            if len(parts) > 1:
                main_text = parts[0].strip()
                sub_queries = parts[1].strip()
                return f"🔍 **{main_text}**\n\n📋 **Sub-queries:**\n{sub_queries}"
            else:
                return f"🔍 **Search**: {content}"
        elif "processing sub-query" in content.lower():
            return f"🔍 **Retrieval**: {content}"
        else:
            return f"🔍 **Search**: {content}"
    elif "📋 **THINKING**:" in step_content:
        return f"📋 **Planning**: {step_content.replace('📋 **THINKING**: ', '').strip()}"
    elif "✅ **THINKING**:" in step_content:
        return f"✅ **Progress**: {step_content.replace('✅ **THINKING**: ', '').strip()}"
    elif "📊 **THINKING**:" in step_content:
        return f"📊 **Analysis**: {step_content.replace('📊 **THINKING**: ', '').strip()}"
    elif "🤖 **THINKING**:" in step_content:
        return f"🤖 **Generation**: {step_content.replace('🤖 **THINKING**: ', '').strip()}"
    elif "⚠️ **THINKING**:" in step_content:
        return f"⚠️ **Warning**: {step_content.replace('⚠️ **THINKING**: ', '').strip()}"
    elif "❌ **THINKING**:" in step_content:
        return f"❌ **Error**: {step_content.replace('❌ **THINKING**: ', '').strip()}"
    else:
        return f"💭 **Thinking**: {step_content}"

def truncate_message(content: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    if len(content) <= max_length:
        return content
    truncate_point = content.rfind('\n---\n', 0, max_length - 100)
    if truncate_point > max_length // 2:
        return content[:truncate_point] + "\n\n... [Content truncated for display] ..."
    else:
        return content[:max_length - 50] + "\n\n... [Content truncated] ..."

# --- Main App ---
def main():
    add_logo_btn1()
    st.sidebar.markdown("""
    ### Instructions:
    1. Ask direct questions for quick facts or analytical questions for deep insights.
    2. April 2024 - March 2025 data available.
    """)
    rag = initialize_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("💬 LumenAI RG Chat")
    st.markdown("Ask me anything about RateGain's financial performance - I'm your CFA!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "thinking_steps" in message:
                with st.expander("🧠 Thinking Process", expanded=False):
                    for step in message["thinking_steps"]:
                        st.markdown(format_thinking_content(step))
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        if isinstance(source, dict):
                            st.markdown(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")
                        else:
                            st.markdown(f"- {source}")

    # Accept user input
    if question := st.chat_input("Ask about RateGain's financial performance..."):
        logging.info(f"User submitted question: {question}")

        # --- Q&A Cache Check ---
        normalized_current = normalize_question(question)
        normalized_cached = normalize_question(st.session_state.last_qa_cache["question"])
        if normalized_cached == normalized_current and st.session_state.last_qa_cache["answer"]:
            logging.info(f"CACHE HIT: '{normalized_current}' == '{normalized_cached}'")
            with st.chat_message("assistant"):
                st.markdown(st.session_state.last_qa_cache["answer"])
                if st.session_state.last_qa_cache["sources"]:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(st.session_state.last_qa_cache["sources"], 1):
                            if isinstance(src, dict):
                                st.markdown(f"**{i}.** {src['file']}, Page: {src['page']} (Relevance: {src['score']:.4f})")
                            else:
                                st.markdown(f"- {src}")
            st.session_state.messages.append({
                "role": "user", "content": question
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": st.session_state.last_qa_cache["answer"],
                "thinking_steps": [],
                "sources": st.session_state.last_qa_cache["sources"]
            })
            st.stop()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            thinking_container = st.container()
            answer_container = st.empty()
            sources_container = st.empty()
            thinking_steps = []
            final_answer = ""
            final_sources = []
            accumulated_thinking = ""
            step_count = 0

            async def run_query():
                nonlocal final_answer, final_sources, accumulated_thinking, step_count
                try:
                    # Create a single expander and a placeholder for its content
                    with thinking_container:
                        with st.expander("🧠 Live Analysis Process", expanded=True):
                            expander_placeholder = st.empty()

                    async for response in rag.query(question, st.session_state.messages):
                        if response["type"] == "thinking":
                            step_count += 1
                            formatted = format_thinking_content(response["content"])
                            thinking_steps.append(response["content"])
                            accumulated_thinking += formatted + "\n\n---\n\n"
                            # Truncate accumulated thinking if too long
                            if len(accumulated_thinking) > MAX_THINKING_LENGTH:
                                trunc_point = accumulated_thinking.find('\n---\n', len(accumulated_thinking) - MAX_THINKING_LENGTH + 1000)
                                if trunc_point > 0:
                                    accumulated_thinking = "...[Earlier steps truncated]...\n\n" + accumulated_thinking[trunc_point:]
                                else:
                                    accumulated_thinking = accumulated_thinking[-MAX_THINKING_LENGTH:]
                            # Update the same expander content
                            expander_placeholder.markdown(
                                accumulated_thinking + f"\n⏳ **Processing... ({step_count} steps completed)**"
                            )
                        elif response["type"] == "answer":
                            final_answer = response["content"]
                            final_sources = response.get("sources", [])
                            # Final update to expander
                            expander_placeholder.markdown(
                                accumulated_thinking + f"\n✅ **Analysis Complete!** ({step_count} steps processed)"
                            )
                            truncated_answer = truncate_message(final_answer, MAX_MESSAGE_LENGTH)
                            answer_container.markdown(truncated_answer)
                            if final_sources:
                                with sources_container.expander("📚 Sources"):
                                    for i, src in enumerate(final_sources, 1):
                                        if isinstance(src, dict):
                                            st.markdown(f"**{i}.** {src['file']}, Page: {src['page']} (Relevance: {src['score']:.4f})")
                                        else:
                                            st.markdown(f"- {src}")
                        elif response["type"] == "error":
                            answer_container.error(f"❌ {response['content']}")
                            logging.error(f"Error response from RAG system: {response['content']}")
                            return
                except Exception as e:
                    answer_container.error(f"❌ Error: {str(e)}")
                    logging.error(f"Exception in run_query: {e}")

            asyncio.run(run_query())

            # Update Q&A cache
            st.session_state.last_qa_cache = {
                "question": normalize_question(question),
                "answer": final_answer,
                "sources": final_sources
            }
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "thinking_steps": thinking_steps,
                "sources": final_sources
            })
            logging.info(f"Assistant response generated: {final_answer[:200]}...")

    # Clear chat button in sidebar
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_qa_cache = {"question": None, "answer": None, "sources": None}
        st.rerun()

    # System status in sidebar
    st.sidebar.markdown("---")
    if rag.is_ready():
        st.sidebar.success("✅ CFA System Ready")
    else:
        st.sidebar.error("❌ System Not Ready")

if __name__ == "__main__":
    main()