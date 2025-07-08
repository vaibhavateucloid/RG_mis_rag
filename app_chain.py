import chainlit as cl
from rag_main import EnhancedRAGSystem
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize RAG system
rag_system = None

def initialize_rag():
    """Initialize the RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = EnhancedRAGSystem()
    return rag_system

def format_thinking_content(step_content):
    """Format thinking content for better display."""
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

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    logging.info("Chat session started.")
    
    # Initialize RAG system
    rag = initialize_rag()
    
    # Store RAG system in session
    cl.user_session.set("rag", rag)
    
    # Initialize chat history as a list of dicts
    cl.user_session.set("chat_history", [])
    
    # Check system status
    if rag.is_ready():
        await cl.Message(
            content="✅ **CFA System Ready**\n\n💡 *Ask for detailed analysis to see the live thinking process!*",
            author="System"
        ).send()
    else:
        await cl.Message(
            content="❌ **System Not Ready**\n\nThere seems to be an issue with the system. Please try again later.",
            author="System"
        ).send()

@cl.step(type="retrieval")
async def retrieve_sources(sources):
    """Process and format sources."""
    if not sources:
        return "No sources found."
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        formatted_sources.append(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")
    
    sources_text = "\n".join(formatted_sources)
    await asyncio.sleep(0.1)
    return sources_text

@cl.on_message
async def main(message: cl.Message):
    """Main message handler with real-time thinking display using message updates."""
    logging.info(f"User submitted question: {message.content}")
    
    # Get RAG system from session
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ System not initialized. Please refresh the page.").send()
        return
    
    # Get chat history (list of dicts)
    chat_history = cl.user_session.get("chat_history", [])
    logging.info(f"Current chat_history: {chat_history}")
    
    # Append user message to chat history
    chat_history.append({"role": "user", "content": message.content})
    logging.info(f"Appended user message. Updated chat_history: {chat_history}")
    
    # Enhanced question with context for follow-ups
    enhanced_question = message.content
    logging.info(f"Passing to rag.query: enhanced_question={enhanced_question}, chat_history={chat_history}")
    
    # Variables to track the process
    final_answer = ""
    final_sources = []
    thinking_steps = []
    step_count = 0
    thinking_message = None
    is_cfa_analysis = False
    
    try:
        # Process query with real-time thinking display
        logging.info("About to enter async for loop over rag.query result...")
        
        async for response in rag.query(enhanced_question, chat_history):
            logging.info(f"Received response from rag.query: {response}")
            
            if response["type"] == "thinking":
                is_cfa_analysis = True
                step_count += 1
                formatted_thinking = format_thinking_content(response["content"])
                thinking_steps.append(formatted_thinking)
                
                # Create thinking message on first thinking step
                if thinking_message is None:
                    thinking_message = cl.Message(
                        content="🧠 **Live Analysis Process**\n\n🔄 Starting deep analysis...",
                        author="CFA Agent"
                    )
                    await thinking_message.send()
                
                # Update the thinking message with all steps so far
                thinking_content = "\n\n---\n\n".join(thinking_steps)
                updated_content = f"🧠 **Live Analysis Process**\n\n{thinking_content}\n\n⏳ **Processing... ({step_count} steps completed)**"
                
                # Update the message content
                thinking_message.content = updated_content
                await thinking_message.update()
                
                # Small delay for better UX
                await asyncio.sleep(0.2)
                
            elif response["type"] == "answer":
                final_answer = response["content"]
                final_sources = response.get("sources", [])
                
                # Update thinking message to show completion
                if thinking_message and is_cfa_analysis:
                    thinking_content = "\n\n---\n\n".join(thinking_steps)
                    completed_content = f"🧠 **Analysis Process Completed**\n\n{thinking_content}\n\n✅ **Analysis Complete!** ({step_count} steps processed)"
                    thinking_message.content = completed_content
                    await thinking_message.update()
                
                # Append assistant response to chat history
                chat_history.append({"role": "assistant", "content": final_answer})
                cl.user_session.set("chat_history", chat_history)
                
                logging.info(f"Assistant response generated: {final_answer}")
                break  # Exit the loop once we get the answer
                
            elif response["type"] == "error":
                error_msg = response["content"]
                logging.error(f"Error response from RAG system: {error_msg}")
                
                # Update thinking message with error if it exists
                if thinking_message:
                    thinking_content = "\n\n---\n\n".join(thinking_steps) if thinking_steps else "Analysis started..."
                    error_content = f"🧠 **Analysis Process**\n\n{thinking_content}\n\n❌ **Error occurred**: {error_msg}"
                    thinking_message.content = error_content
                    await thinking_message.update()
                
                # Send error message and return
                await cl.Message(content=f"❌ **Error**: {error_msg}").send()
                return
        
        logging.info("Exited async for loop over rag.query result.")
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(f"Exception in main: {error_msg}")
        
        # Update thinking message with exception if it exists
        if thinking_message:
            thinking_content = "\n\n---\n\n".join(thinking_steps) if thinking_steps else "Analysis started..."
            exception_content = f"🧠 **Analysis Process**\n\n{thinking_content}\n\n❌ **Exception occurred**: {error_msg}"
            thinking_message.content = exception_content
            await thinking_message.update()
        
        # Send error message and return
        await cl.Message(content=f"❌ **Error**: {error_msg}").send()
        return
    
    # Add delay before sending final answer
    await asyncio.sleep(0.3)
    
    # Send the final answer
    if final_answer:
        try:
            # Create the final answer message
            await cl.Message(content=final_answer).send()
            logging.info("Final message sent successfully to UI")
            
            # Add delay before sources
            await asyncio.sleep(0.2)
            
            # Send sources as a separate collapsible step if available
            if final_sources:
                async with cl.Step(name="📚 Sources", type="retrieval", show_input=False) as sources_step:
                    sources_text = await retrieve_sources(final_sources)
                    sources_step.output = f"**Retrieved {len(final_sources)} sources:**\n\n{sources_text}"
                    
        except Exception as e:
            logging.error(f"Error sending final message: {e}")
            await cl.Message(content=f"❌ Error displaying results: {str(e)}").send()
    else:
        logging.warning("No final answer to send")
        await cl.Message(content="❌ No response generated").send()

@cl.on_chat_end
async def end():
    """Handle chat session end."""
    logging.info("Chat session ended.")

if __name__ == "__main__":
    # This is handled by chainlit run command
    pass