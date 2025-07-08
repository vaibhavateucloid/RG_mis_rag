# rag_adapter.py - Async adapter for RAG system to work with Chainlit

import asyncio
from typing import AsyncGenerator, Dict, Any
from rag_main import RAGSystem
import logging

class AsyncRAGAdapter:
    """
    Async adapter for RAG system to work with Chainlit.
    This wraps the existing RAG system and makes it async-compatible.
    """
    
    def __init__(self):
        self.rag_system = RAGSystem()
        
    def is_ready(self) -> bool:
        """Check if the RAG system is ready."""
        return self.rag_system.is_ready()
    
    async def query(self, question: str, conversation_context: str = "") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator that yields responses from the RAG system.
        
        Args:
            question: The user's question
            conversation_context: Previous conversation context
            
        Yields:
            Dict with 'type' and 'content' keys
        """
        try:
            # Run the synchronous RAG query in a thread pool
            loop = asyncio.get_event_loop()
            
            # Create a generator from the sync RAG system
            def sync_generator():
                return self.rag_system.query(question, conversation_context)
            
            # Use an asyncio.Queue to pass results from the sync generator to the async generator
            queue = asyncio.Queue()

            def run_sync_generator():
                try:
                    for response in self.rag_system.query(question, conversation_context):
                        # Put the response into the queue
                        loop.call_soon_threadsafe(queue.put_nowait, response)
                except Exception as e:
                    logging.error(f"Error in sync_generator thread: {str(e)}")
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": f"Internal error: {str(e)}"})
                finally:
                    # Signal that the generator is done
                    loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel value

            # Run the synchronous generator in a separate thread
            loop.run_in_executor(None, run_sync_generator)

            # Asynchronously yield responses from the queue
            while True:
                response = await queue.get()
                if response is None:  # Sentinel value indicates end of generator
                    break
                yield response
                await asyncio.sleep(0.01) # Small delay to allow for UI updates
                
        except Exception as e:
            logging.error(f"Error in AsyncRAGAdapter.query: {str(e)}")
            yield {
                "type": "error",
                "content": f"An error occurred while processing your query: {str(e)}"
            }

# Alternative: If you want to modify the original RAG system directly
class ModifiedRAGSystem(RAGSystem):
    """
    Modified version of RAGSystem that works with async/await.
    You can use this if you want to modify the original RAG system directly.
    """
    
    async def async_query(self, question: str, conversation_context: str = "") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async version of the query method.
        """
        try:
            # If your original RAG system uses generators, wrap them:
            for response in self.query(question, conversation_context):
                yield response
                await asyncio.sleep(0.01)  # Allow UI updates
                
        except Exception as e:
            logging.error(f"Error in ModifiedRAGSystem.async_query: {str(e)}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}"
            }