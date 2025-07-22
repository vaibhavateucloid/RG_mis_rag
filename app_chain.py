# Enhanced Chainlit App with Collapsible Live Thinking Display
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
   
    # Show usage instructions on first access
    await cl.Message(
        content=(
            "⚠️ **Notice:** This is an under-development app.\n\n"
            "- Please report any errors immediately.\n"
            "- Refresh the app periodically.\n"
            "- For best results, open a new chat after every 4-5 messages.\n"
            "- If you see a wrong response or an error in displaying the message, refresh the page.\n"
            "\nThank you for helping us improve!"
        ),
        author="System"
    ).send()

    # Initialize RAG system
    rag = initialize_rag()
   
    # Store RAG system in session
    cl.user_session.set("rag", rag)
   
    # Initialize chat history as a list of dicts
    cl.user_session.set("chat_history", [])
   
    # Check system status
    if rag.is_ready():
        await cl.Message(
            content="✅ **LumenAI RG-MIS Chatbot Ready...**\n\n💡 *Ask for detailed analysis to see the live thinking process!*\n\n🔽 *Click the dropdown arrow on thinking steps to see real-time analysis*",
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
    """Main message handler with immediate message sending."""
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
    step_count = 0
    thinking_step = None
    is_cfa_analysis = False
    accumulated_thinking = ""
   
    try:
        # Process query with real-time thinking display
        logging.info("About to enter async for loop over rag.query result...")
       
        async for response in rag.query(enhanced_question, chat_history):
            logging.info(f"Received response from rag.query: {response}")
           
            if response["type"] == "thinking":
                is_cfa_analysis = True
                step_count += 1
                formatted_thinking = format_thinking_content(response["content"])
               
                # Create thinking step on first thinking response
                if thinking_step is None:
                    thinking_step = cl.Step(
                        name=f"🧠 Live Analysis Process (Click to expand)",
                        type="tool",
                        show_input=False
                    )
                    await thinking_step.__aenter__()
                    thinking_step.input = "🔄 Starting deep CFA analysis..."
                    accumulated_thinking = "🔄 **Initializing Analysis**\n\nStarting comprehensive financial analysis process...\n\n---\n\n"
               
                # Add new thinking step to accumulated content
                accumulated_thinking += f"{formatted_thinking}\n\n---\n\n"
               
                # Update the step output with accumulated thinking
                thinking_step.output = f"{accumulated_thinking}⏳ **Processing... ({step_count} steps completed)**"
               
                # Stream the new content token by token for better UX
                new_content = f"{formatted_thinking}\n\n---\n\n"
                for char in new_content:
                    await thinking_step.stream_token(char)
                    await asyncio.sleep(0.01)
               
                # Update step name to show progress
                thinking_step.name = f"🧠 Live Analysis Process ({step_count} steps) - Click to expand"
                await thinking_step.update()
               
            elif response["type"] == "answer":
                final_answer = response["content"]
                final_sources = response.get("sources", [])
               
                # ⚡ SEND MESSAGE IMMEDIATELY - NO DELAYS
                logging.info("🚀 DEBUG: Sending message IMMEDIATELY to prevent session timeout")
                
                # Check session is still active
                try:
                    session_id = cl.user_session.get("id", "unknown")
                    logging.info(f"🔍 DEBUG: Session still active: {session_id}")
                except Exception as e:
                    logging.error(f"❌ DEBUG: Session access error: {e}")
                
                # Send final answer FIRST, before any other operations
                try:
                    await cl.Message(content=final_answer).send()
                    logging.info("✅ DEBUG: Final message sent IMMEDIATELY")
                except Exception as e:
                    logging.error(f"❌ DEBUG: Immediate message send failed: {e}")
               
                # Complete the thinking step AFTER message is sent
                if thinking_step and is_cfa_analysis:
                    thinking_step.output = f"{accumulated_thinking}✅ **Analysis Complete!** ({step_count} steps processed)\n\n🎯 **Comprehensive answer provided**"
                    thinking_step.name = f"🧠 Analysis Complete ({step_count} steps) - Click to expand"
                    await thinking_step.update()
               
                # Append assistant response to chat history
                chat_history.append({"role": "assistant", "content": final_answer})
                cl.user_session.set("chat_history", chat_history)
               
                logging.info(f"Assistant response generated and sent immediately")
                break
               
            elif response["type"] == "error":
                error_msg = response["content"]
                logging.error(f"Error response from RAG system: {error_msg}")
               
                # Send error immediately
                await cl.Message(content=f"❌ **Error**: {error_msg}").send()
               
                if thinking_step:
                    thinking_step.output = f"{accumulated_thinking}❌ **Error occurred**: {error_msg}"
                    thinking_step.name = f"🧠 Analysis Error - Click to expand"
                    await thinking_step.update()
               
                return
       
        logging.info("Exited async for loop over rag.query result.")
       
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(f"Exception in main: {error_msg}")
       
        # Send error immediately
        await cl.Message(content=f"❌ **Error**: {error_msg}").send()
        
        if thinking_step:
            thinking_step.output = f"{accumulated_thinking}❌ **Exception occurred**: {error_msg}"
            thinking_step.name = f"🧠 Analysis Exception - Click to expand"
            await thinking_step.update()
       
        return
   
    # Send sources AFTER main message (if session still active)
    if final_sources:
        try:
            await asyncio.sleep(0.2)  # Small delay only for sources
            async with cl.Step(name="📚 Sources", type="retrieval", show_input=False) as sources_step:
                sources_text = await retrieve_sources(final_sources)
                sources_step.output = f"**Retrieved {len(final_sources)} sources:**\n\n{sources_text}"
                logging.info("✅ DEBUG: Sources sent after main message")
        except Exception as e:
            logging.error(f"❌ DEBUG: Sources failed (session may have ended): {e}")
    
    logging.info("🔍 DEBUG: Main function execution completed")
 
@cl.on_chat_end
async def end():
    """Handle chat session end."""
    logging.info("Chat session ended.")
 
if __name__ == "__main__":
    # This is handled by chainlit run command
    pass