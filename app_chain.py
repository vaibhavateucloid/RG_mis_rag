# # Enhanced Chainlit App with Live Thinking Display

# import chainlit as cl
# from rag_adapter import AsyncRAGAdapter # Using the async adapter for RAG system
# import logging
# import asyncio

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# # Initialize RAG system
# rag_system = None

# def initialize_rag():
#     """Initialize the RAG system."""
#     global rag_system
#     if rag_system is None:
#         rag_system = AsyncRAGAdapter()
#     return rag_system

# def format_thinking_step(step_content):
#     """Format a thinking step with appropriate emoji and styling."""
#     if "🧠 **THINKING**:" in step_content:
#         step_text = step_content.replace("🧠 **THINKING**: ", "").strip()
#         return f"🧠 Starting analysis: {step_text}"
#     elif "🔍 **THINKING**:" in step_content:
#         step_text = step_content.replace("🔍 **THINKING**: ", "").strip()
#         return f"🔍 Searching: {step_text}"
#     elif "📋 **THINKING**:" in step_content:
#         step_text = step_content.replace("📋 **THINKING**: ", "").strip()
#         if "sub-queries:" in step_text:
#             parts = step_text.split("sub-queries:")
#             if len(parts) > 1:
#                 return f"📋 Planning: {parts[0].strip()}\n\nSub-queries:\n{parts[1].strip()}"
#         return f"📋 Planning: {step_text}"
#     elif "✅ **THINKING**:" in step_content:
#         step_text = step_content.replace("✅ **THINKING**: ", "").strip()
#         return f"✅ Progress: {step_text}"
#     elif "📊 **THINKING**:" in step_content:
#         step_text = step_content.replace("📊 **THINKING**: ", "").strip()
#         return f"📊 Analyzing: {step_text}"
#     elif "🤖 **THINKING**:" in step_content:
#         step_text = step_content.replace("🤖 **THINKING**: ", "").strip()
#         return f"🤖 Generating: {step_text}"
#     elif "⚠️ **THINKING**:" in step_content:
#         step_text = step_content.replace("⚠️ **THINKING**: ", "").strip()
#         return f"⚠️ Warning: {step_text}"
#     elif "❌ **THINKING**:" in step_content:
#         step_text = step_content.replace("❌ **THINKING**: ", "").strip()
#         return f"❌ Error: {step_text}"
#     else:
#         return step_content

# @cl.on_chat_start
# async def start():
#     """Initialize the chat session."""
#     logging.info("Chat session started.")
    
#     # Initialize RAG system
#     rag = initialize_rag()
    
#     # Store RAG system in session
#     cl.user_session.set("rag", rag)
    
#     # Initialize conversation context
#     cl.user_session.set("conversation_context", "")
    
#     # Check system status
#     if rag.is_ready():
#         await cl.Message(
#             content="✅ **CFA System Ready**",
#             author="System"
#         ).send()
#     else:
#         await cl.Message(
#             content="❌ **System Not Ready**\n\nThere seems to be an issue with the system. Please try again later.",
#             author="System"
#         ).send()

# @cl.step(type="tool")
# async def process_thinking_step(step_content: str):
#     """Process a thinking step and return formatted content."""
#     formatted_content = format_thinking_step(step_content)
#     await asyncio.sleep(0.1)  # Small delay to simulate processing
#     return formatted_content

# @cl.step(type="retrieval")
# async def retrieve_sources(sources):
#     """Process and format sources."""
#     if not sources:
#         return "No sources found."
    
#     formatted_sources = []
#     for i, source in enumerate(sources, 1):
#         formatted_sources.append(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")
    
#     sources_text = "\n".join(formatted_sources)
#     await asyncio.sleep(0.1)  # Small delay to simulate processing
#     return sources_text

# @cl.on_message
# async def main(message: cl.Message):
#     """Main message handler with live thinking display."""
#     logging.info(f"User submitted question: {message.content}")
    
#     # Get RAG system from session
#     rag = cl.user_session.get("rag")
#     if not rag:
#         await cl.Message(content="❌ System not initialized. Please refresh the page.").send()
#         return
    
#     # Get conversation context
#     conversation_context = cl.user_session.get("conversation_context", "")
    
#     # Update conversation context
#     conversation_context += f"user: {message.content}\n"
    
#     # Enhanced question with context for follow-ups
#     enhanced_question = f"Previous conversation:\n{conversation_context}\nCurrent question: {message.content}" if len(conversation_context) > 50 else message.content
    
#     # Initialize response message
#     response_msg = cl.Message(content="🧠 **Starting analysis...**")
#     await response_msg.send()
    
#     # Variables to track the process
#     thinking_steps = []
#     final_answer = ""
#     final_sources = []
    
#     try:
#         # Process query with live thinking display
#         async for response in rag.query(enhanced_question, conversation_context):
#             if response["type"] == "thinking":
#                 # Process thinking step
#                 thinking_step = await process_thinking_step(response["content"])
#                 thinking_steps.append(thinking_step)
                
#                 # Update message with current progress
#                 thinking_display = "\n\n".join(thinking_steps)
#                 current_content = f"🧠 **Live Thinking Process**\n\n{thinking_display}\n\n⏳ **Analyzing and generating answer...**"
#                 response_msg.content = current_content
#                 await response_msg.update()
                
#             elif response["type"] == "answer":
#                 final_answer = response["content"]
#                 final_sources = response.get("sources", [])
                
#                 # Process sources
#                 sources_text = ""
#                 if final_sources:
#                     sources_text = await retrieve_sources(final_sources)
                
#                 # Update conversation context
#                 conversation_context += f"assistant: {final_answer}\n"
#                 cl.user_session.set("conversation_context", conversation_context)
                
#                 # Final response
#                 response_msg.content = final_answer
#                 await response_msg.update()
                
#                 # Send sources as a separate message if available
#                 # if sources_text and sources_text != "No sources found.":
#                 #     sources_msg = cl.Message(
#                 #         content=f"📚 **Sources www**\n\n{sources_text}",
#                 #         author="Sources"
#                 #     )
#                 #     await sources_msg.send()
                
#                 logging.info(f"Assistant response generated: {final_answer}")
                
#             elif response["type"] == "error":
#                 error_msg = response["content"]
#                 logging.error(f"Error response from RAG system: {error_msg}")
#                 response_msg.content = f"❌ **Error**: {error_msg}"
#                 await response_msg.update()
                
#     except Exception as e:
#         error_msg = f"An error occurred: {str(e)}"
#         logging.error(f"Exception in main: {error_msg}")
#         response_msg.content = f"❌ **Error**: {error_msg}"
#         await response_msg.update()

# @cl.on_chat_end
# async def end():
#     """Handle chat session end."""
#     logging.info("Chat session ended.")

# if __name__ == "__main__":
#     # This is handled by chainlit run command
#     pass

# Enhanced Chainlit App with Collapsible Live Thinking Display

# import chainlit as cl
# from rag_adapter import AsyncRAGAdapter # Using the async adapter for RAG system
# import logging
# import asyncio

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# # Initialize RAG system
# rag_system = None

# def initialize_rag():
#     """Initialize the RAG system."""
#     global rag_system
#     if rag_system is None:
#         rag_system = AsyncRAGAdapter()
#     return rag_system

# def get_step_type_and_name(step_content):
#     """Determine step type and name based on content."""
#     if "🧠 **THINKING**:" in step_content:
#         return "llm", "🧠 Analysis", step_content.replace("🧠 **THINKING**: ", "").strip()
#     elif "🔍 **THINKING**:" in step_content:
#         return "retrieval", "🔍 Search", step_content.replace("🔍 **THINKING**: ", "").strip()
#     elif "📋 **THINKING**:" in step_content:
#         return "tool", "📋 Planning", step_content.replace("📋 **THINKING**: ", "").strip()
#     elif "✅ **THINKING**:" in step_content:
#         return "tool", "✅ Progress", step_content.replace("✅ **THINKING**: ", "").strip()
#     elif "📊 **THINKING**:" in step_content:
#         return "tool", "📊 Analysis", step_content.replace("📊 **THINKING**: ", "").strip()
#     elif "🤖 **THINKING**:" in step_content:
#         return "llm", "🤖 Generation", step_content.replace("🤖 **THINKING**: ", "").strip()
#     elif "⚠️ **THINKING**:" in step_content:
#         return "tool", "⚠️ Warning", step_content.replace("⚠️ **THINKING**: ", "").strip()
#     elif "❌ **THINKING**:" in step_content:
#         return "tool", "❌ Error", step_content.replace("❌ **THINKING**: ", "").strip()
#     else:
#         return "tool", "🤔 Thinking", step_content

# @cl.on_chat_start
# async def start():
#     """Initialize the chat session."""
#     logging.info("Chat session started.")
    
#     # Initialize RAG system
#     rag = initialize_rag()
    
#     # Store RAG system in session
#     cl.user_session.set("rag", rag)
    
#     # Initialize conversation context
#     cl.user_session.set("conversation_context", "")
    
#     # Check system status
#     if rag.is_ready():
#         await cl.Message(
#             content="✅ **CFA System Ready**",
#             author="System"
#         ).send()
#     else:
#         await cl.Message(
#             content="❌ **System Not Ready**\n\nThere seems to be an issue with the system. Please try again later.",
#             author="System"
#         ).send()

# @cl.step(type="retrieval")
# async def retrieve_sources(sources):
#     """Process and format sources."""
#     if not sources:
#         return "No sources found."
    
#     formatted_sources = []
#     for i, source in enumerate(sources, 1):
#         formatted_sources.append(f"**{i}.** {source['file']}, Page: {source['page']} (Relevance: {source['score']:.4f})")
    
#     sources_text = "\n".join(formatted_sources)
#     await asyncio.sleep(0.1)  # Small delay to simulate processing
#     return sources_text

# @cl.on_message
# async def main(message: cl.Message):
#     """Main message handler with collapsible live thinking display."""
#     logging.info(f"User submitted question: {message.content}")
    
#     # Get RAG system from session
#     rag = cl.user_session.get("rag")
#     if not rag:
#         await cl.Message(content="❌ System not initialized. Please refresh the page.").send()
#         return
    
#     # Get conversation context
#     conversation_context = cl.user_session.get("conversation_context", "")
    
#     # Update conversation context
#     conversation_context += f"user: {message.content}\n"
    
#     # Enhanced question with context for follow-ups
#     enhanced_question = f"Previous conversation:\n{conversation_context}\nCurrent question: {message.content}" if len(conversation_context) > 50 else message.content
    
#     # Create a parent step for the entire thinking process
#     async with cl.Step(name="🧠 Live Thinking Process", type="llm") as parent_step:
#         parent_step.input = message.content
        
#         # Variables to track the process
#         thinking_steps = []
#         final_answer = ""
#         final_sources = []
#         current_thinking_step = None
        
#         try:
#             # Process query with live thinking display
#             async for response in rag.query(enhanced_question, conversation_context):
#                 if response["type"] == "thinking":
#                     step_type, step_name, step_content = get_step_type_and_name(response["content"])
                    
#                     # Create a new thinking step
#                     async with cl.Step(name=step_name, type=step_type) as thinking_step:
#                         thinking_step.input = "Processing..."
                        
#                         # Handle sub-queries formatting
#                         if "sub-queries:" in step_content:
#                             parts = step_content.split("sub-queries:")
#                             if len(parts) > 1:
#                                 formatted_content = f"{parts[0].strip()}\n\n**Sub-queries:**\n{parts[1].strip()}"
#                                 thinking_step.output = formatted_content
#                             else:
#                                 thinking_step.output = step_content
#                         else:
#                             thinking_step.output = step_content
                        
#                         thinking_steps.append(step_name)
                        
#                         # Small delay to show the step
#                         await asyncio.sleep(0.1)
                    
#                 elif response["type"] == "answer":
#                     final_answer = response["content"]
#                     final_sources = response.get("sources", [])
                    
#                     # Update conversation context
#                     conversation_context += f"assistant: {final_answer}\n"
#                     cl.user_session.set("conversation_context", conversation_context)
                    
#                     # Set the parent step output
#                     steps_summary = f"Completed {len(thinking_steps)} thinking steps: {', '.join(thinking_steps)}"
#                     parent_step.output = steps_summary
                    
#                     logging.info(f"Assistant response generated: {final_answer}")
                    
#                 elif response["type"] == "error":
#                     error_msg = response["content"]
#                     logging.error(f"Error response from RAG system: {error_msg}")
                    
#                     # Create an error step
#                     async with cl.Step(name="❌ Error", type="tool") as error_step:
#                         error_step.input = "Error occurred"
#                         error_step.output = error_msg
                    
#                     parent_step.output = f"Error: {error_msg}"
                    
#                     # Send error message
#                     await cl.Message(content=f"❌ **Error**: {error_msg}").send()
#                     return
                    
#         except Exception as e:
#             error_msg = f"An error occurred: {str(e)}"
#             logging.error(f"Exception in main: {error_msg}")
            
#             # Create an exception step
#             async with cl.Step(name="❌ Exception", type="tool") as exception_step:
#                 exception_step.input = "Exception occurred"
#                 exception_step.output = error_msg
            
#             parent_step.output = f"Exception: {error_msg}"
            
#             # Send error message
#             await cl.Message(content=f"❌ **Error**: {error_msg}").send()
#             return
    
#     # Send the final answer
#     if final_answer:
#         await cl.Message(content=final_answer).send()
        
#         # Send sources as a separate collapsible step if available
#         if final_sources:
#             async with cl.Step(name="📚 Sources", type="retrieval") as sources_step:
#                 sources_step.input = f"Retrieved {len(final_sources)} sources"
#                 sources_text = await retrieve_sources(final_sources)
#                 sources_step.output = sources_text

# @cl.on_chat_end
# async def end():
#     """Handle chat session end."""
#     logging.info("Chat session ended.")

# if __name__ == "__main__":
#     # This is handled by chainlit run command
#     pass

# Enhanced Chainlit App with Collapsible Live Thinking Display
# Enhanced Chainlit App with Real-Time Collapsible Thinking Display

import chainlit as cl
from rag_adapter import AsyncRAGAdapter # Using the async adapter for RAG system
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
        rag_system = AsyncRAGAdapter()
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
                return f"🔍 **{main_text}**\n📋 **Sub-queries:**\n{sub_queries}"
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
    
    # Initialize conversation context
    cl.user_session.set("conversation_context", "")
    
    # Check system status
    if rag.is_ready():
        await cl.Message(
            content="✅ **CFA System Ready**",
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
    await asyncio.sleep(0.1)  # Small delay to simulate processing
    return sources_text

@cl.on_message
async def main(message: cl.Message):
    """Main message handler with real-time thinking stream."""
    logging.info(f"User submitted question: {message.content}")
    
    # Get RAG system from session
    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="❌ System not initialized. Please refresh the page.").send()
        return
    
    # Get conversation context
    conversation_context = cl.user_session.get("conversation_context", "")
    
    # Update conversation context
    conversation_context += f"user: {message.content}\n"
    
    # Enhanced question with context for follow-ups
    enhanced_question = f"Previous conversation:\n{conversation_context}\nCurrent question: {message.content}" if len(conversation_context) > 50 else message.content
    
    # Variables to track the process
    final_answer = ""
    final_sources = []
    thinking_content = ""
    step_count = 0
    
    # Create a single step for the entire thinking process
    async with cl.Step(name="🧠 Live Analysis Process", type="llm", show_input=False) as thinking_step:
        thinking_step.input = f"Analyzing: {message.content}"
        
        try:
            # Process query with real-time thinking display
            async for response in rag.query(enhanced_question, conversation_context):
                if response["type"] == "thinking":
                    step_count += 1
                    formatted_thinking = format_thinking_content(response["content"])
                    
                    # Add to accumulated thinking content
                    if thinking_content:
                        thinking_content += f"\n\n---\n\n{formatted_thinking}"
                    else:
                        thinking_content = formatted_thinking
                    
                    # Update the step output in real-time
                    thinking_step.output = f"{thinking_content}\n\n**Steps processed: {step_count}**"
                    
                    # Small delay for better UX
                    await asyncio.sleep(0.05)
                    
                elif response["type"] == "answer":
                    final_answer = response["content"]
                    final_sources = response.get("sources", [])
                    
                    # Update conversation context
                    conversation_context += f"assistant: {final_answer}\n"
                    cl.user_session.set("conversation_context", conversation_context)
                    
                    # Set final step output
                    thinking_step.output = f"{thinking_content}\n\n✅ **Analysis Complete!** ({step_count} steps processed)"
                    
                    logging.info(f"Assistant response generated: {final_answer}")
                    
                elif response["type"] == "error":
                    error_msg = response["content"]
                    logging.error(f"Error response from RAG system: {error_msg}")
                    
                    # Update step with error
                    thinking_step.output = f"{thinking_content}\n\n❌ **Error occurred**: {error_msg}"
                    
                    # Send error message
                    await cl.Message(content=f"❌ **Error**: {error_msg}").send()
                    return
                    
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logging.error(f"Exception in main: {error_msg}")
            
            # Update step with exception
            thinking_step.output = f"{thinking_content}\n\n❌ **Exception occurred**: {error_msg}"
            
            # Send error message
            await cl.Message(content=f"❌ **Error**: {error_msg}").send()
            return
    
    # Send the final answer
    if final_answer:
        await cl.Message(content=final_answer).send()
        
        # Send sources as a separate collapsible step if available
        if final_sources:
            async with cl.Step(name="📚 Sources", type="retrieval", show_input=False) as sources_step:
                sources_text = await retrieve_sources(final_sources)
                sources_step.output = f"**Retrieved {len(final_sources)} sources:**\n\n{sources_text}"

@cl.on_chat_end
async def end():
    """Handle chat session end."""
    logging.info("Chat session ended.")

if __name__ == "__main__":
    # This is handled by chainlit run command
    pass