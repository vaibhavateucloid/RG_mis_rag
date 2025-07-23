# Enhanced Chainlit App with Collapsible Live Thinking Display - FIXED VERSION
import chainlit as cl
from rag_main import EnhancedRAGSystem
import logging
import asyncio
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
 
# Initialize RAG system
rag_system = None

# Constants for message handling
MAX_MESSAGE_LENGTH = 50000  # Chainlit message length limit
MAX_THINKING_LENGTH = 20000  # Limit accumulated thinking to prevent memory issues
 
def initialize_rag():
    """Initialize the RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = EnhancedRAGSystem()
    return rag_system

def truncate_message(content: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Truncate message if it exceeds length limit."""
    if len(content) <= max_length:
        return content
    
    # Try to truncate at a logical point (end of section)
    truncate_point = content.rfind('\n---\n', 0, max_length - 100)
    if truncate_point > max_length // 2:  # Only if we found a good truncation point
        return content[:truncate_point] + "\n\n... [Content truncated for display] ..."
    else:
        return content[:max_length - 50] + "\n\n... [Content truncated] ..."

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
    """Main message handler with collapsible real-time thinking display - FIXED VERSION."""
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
    max_steps_before_summary = 15  # Prevent too many steps
   
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
                        show_input=False  # This makes it collapsible and closed by default
                    )
                    await thinking_step.__aenter__()
                    thinking_step.input = "🔄 Starting deep CFA analysis..."
                    accumulated_thinking = "🔄 **Initializing Analysis**\n\nStarting comprehensive financial analysis process...\n\n---\n\n"
               
                # Add new thinking step to accumulated content with length control
                new_step_content = f"{formatted_thinking}\n\n---\n\n"
                accumulated_thinking += new_step_content
                
                # Keep accumulated thinking under control
                if len(accumulated_thinking) > MAX_THINKING_LENGTH:
                    # Keep only the last portion and add a note about truncation
                    truncation_point = accumulated_thinking.find('\n---\n', len(accumulated_thinking) - MAX_THINKING_LENGTH + 1000)
                    if truncation_point > 0:
                        accumulated_thinking = "...[Earlier steps truncated]...\n\n" + accumulated_thinking[truncation_point:]
                    else:
                        accumulated_thinking = accumulated_thinking[-MAX_THINKING_LENGTH:] 
               
                # Update the step output with accumulated thinking
                progress_text = f"⏳ **Processing... ({step_count} steps completed)**"
                if step_count > max_steps_before_summary:
                    progress_text = f"⏳ **Processing... ({step_count} steps completed - wrapping up)**"
                
                thinking_step.output = f"{accumulated_thinking}{progress_text}"
               
                # FIXED: Simplified streaming - no character-by-character streaming
                # Just update the step content in larger chunks to avoid race conditions
                await thinking_step.update()
               
                # Update step name to show progress
                thinking_step.name = f"🧠 Live Analysis Process ({step_count} steps) - Click to expand"
               
            elif response["type"] == "answer":
                final_answer = response["content"]
                final_sources = response.get("sources", [])
               
                # Complete the thinking step
                if thinking_step and is_cfa_analysis:
                    thinking_step.output = f"{accumulated_thinking}✅ **Analysis Complete!** ({step_count} steps processed)\n\n🎯 **Ready to provide comprehensive answer**"
                    thinking_step.name = f"🧠 Analysis Complete ({step_count} steps) - Click to expand"
                    await thinking_step.update()
                    await thinking_step.__aexit__(None, None, None)
               
                # Append assistant response to chat history
                chat_history.append({"role": "assistant", "content": final_answer})
                cl.user_session.set("chat_history", chat_history)
               
                logging.info(f"Assistant response generated: {final_answer[:200]}... (length: {len(final_answer)})")
                break  # Exit the loop once we get the answer
               
            elif response["type"] == "error":
                error_msg = response["content"]
                logging.error(f"Error response from RAG system: {error_msg}")
               
                # Update thinking step with error if it exists
                if thinking_step:
                    thinking_step.output = f"{accumulated_thinking}❌ **Error occurred**: {error_msg}"
                    thinking_step.name = f"🧠 Analysis Error - Click to expand"
                    await thinking_step.update()
                    await thinking_step.__aexit__(None, None, None)
               
                # Send error message and return
                await cl.Message(content=f"❌ **Error**: {error_msg}").send()
                return
       
        logging.info("Exited async for loop over rag.query result.")
       
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(f"Exception in main: {error_msg}")
       
        # Update thinking step with exception if it exists
        if thinking_step:
            thinking_step.output = f"{accumulated_thinking}❌ **Exception occurred**: {error_msg}"
            thinking_step.name = f"🧠 Analysis Exception - Click to expand"
            await thinking_step.update()
            await thinking_step.__aexit__(None, None, None)
       
        # Send error message and return
        await cl.Message(content=f"❌ **Error**: {error_msg}").send()
        return
   
    # FIXED: Add delay and ensure clean state before sending final answer
    await asyncio.sleep(0.5)  # Give thinking step time to complete
   
    # Send the final answer with improved error handling
    if final_answer:
        try:
            # FIXED: Check and truncate message if too long
            truncated_answer = truncate_message(final_answer, MAX_MESSAGE_LENGTH)
            
            if len(final_answer) != len(truncated_answer):
                logging.warning(f"Message truncated from {len(final_answer)} to {len(truncated_answer)} characters")
            
            # Create the final answer message with timeout protection
            try:
                message_task = asyncio.create_task(cl.Message(content=truncated_answer).send())
                await asyncio.wait_for(message_task, timeout=30.0)  # 30 second timeout
                logging.info("Final message sent successfully to UI")
            except asyncio.TimeoutError:
                logging.error("Timeout sending final message - trying fallback")
                # Fallback: try with much shorter message
                short_answer = final_answer[:5000] + "\n\n... [Message truncated due to display issues. Please ask for specific sections if needed.]"
                await cl.Message(content=short_answer).send()
                logging.info("Fallback short message sent")
           
            # Add delay before sources
            await asyncio.sleep(0.3)
           
            # Send sources as a separate collapsible step if available
            if final_sources:
                try:
                    async with cl.Step(name="📚 Sources", type="retrieval", show_input=False) as sources_step:
                        sources_text = await retrieve_sources(final_sources)
                        sources_step.output = f"**Retrieved {len(final_sources)} sources:**\n\n{sources_text}"
                except Exception as source_error:
                    logging.error(f"Error displaying sources: {source_error}")
                    # Don't fail the whole response for source display issues
                   
        except Exception as e:
            logging.error(f"Error sending final message: {e}")
            # Emergency fallback
            try:
                emergency_msg = f"✅ **Analysis Complete**\n\nAnalysis was completed successfully but there was a display issue. Please try asking for specific parts of the analysis or refresh the page.\n\n**Error details**: {str(e)}"
                await cl.Message(content=emergency_msg).send()
            except Exception as emergency_error:
                logging.error(f"Emergency message also failed: {emergency_error}")
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