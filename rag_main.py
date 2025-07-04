# This is where the main code for retrival and generation is there which will see if any data is there in the vector_store... if data is there it will proceed but if data is not there it will just call the data_processing.py and get the vector store ready and then proceed 


# RAG Main - Retrieval and Generation System
# Checks if vector store exists → If not, calls data_processing.py → Proceeds with RAG

import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
import chromadb

# Import data processing pipeline
from data_processing import main as run_data_processing, Config, GeminiEmbeddings, load_existing_vector_store

# Load environment variables
load_dotenv()

class RAGSystem:
    """Main RAG system that handles both setup and querying."""
    
    def __init__(self, 
                 vector_store_path: str = None,
                 collection_name: str = None,
                 embedding_model: str = "gemini-embedding-exp-03-07",
                 generation_model: str = "gemini-2.5-pro"):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_path: Path to the ChromaDB vector store
            collection_name: Name of the collection in ChromaDB
            embedding_model: Embedding model for query encoding
            generation_model: Generation model for response creation
        """
        # Use Config defaults if not provided
        self.config = Config()
        self.vector_store_path = vector_store_path or self.config.VECTOR_DB_DIR
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.genai_client = None
        
        # Setup the system
        self._setup_system()
    
    def _setup_system(self):
        """Setup the complete RAG system."""
        print("🚀 Initializing RAG System...")
        
        # Step 1: Check if vector store exists
        if not self._vector_store_exists():
            print("📋 Vector store not found. Running data processing pipeline...")
            self._run_data_processing()
        
        # Step 2: Initialize components
        self._initialize_components()
        
        print("✅ RAG System ready!")
    
    def _vector_store_exists(self) -> bool:
        """Check if vector store exists."""
        try:
            if not os.path.exists(self.vector_store_path):
                return False
            
            # Try to connect to ChromaDB
            client = chromadb.PersistentClient(path=self.vector_store_path)
            collections = client.list_collections()
            
            # Check if our collection exists
            collection_names = [col.name for col in collections]
            return self.collection_name in collection_names
            
        except Exception as e:
            print(f"⚠️ Error checking vector store: {str(e)}")
            return False
    
    def _run_data_processing(self):
        """Run the data processing pipeline."""
        try:
            print("🔄 Starting data processing pipeline...")
            documents, chunks, vector_store = run_data_processing()
            
            if not vector_store:
                raise Exception("Data processing failed to create vector store")
            
            print("✅ Data processing completed successfully!")
            
        except Exception as e:
            print(f"❌ Data processing failed: {str(e)}")
            raise
    
    def _initialize_components(self):
        """Initialize RAG components."""
        try:
            # Initialize embedding model
            self.embeddings = GeminiEmbeddings(self.embedding_model)
            print(f"✅ Embedding model loaded: {self.embedding_model}")
            
            # Initialize vector store
            self.vector_store = load_existing_vector_store(
                persist_directory=self.vector_store_path,
                embedding_model_name=self.embedding_model,
                collection_name=self.collection_name
            )
            
            if not self.vector_store:
                raise Exception("Failed to load vector store")
            
            # Initialize generation client
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            
            self.genai_client = genai.Client(api_key=api_key)
            print(f"✅ Generation model initialized: {self.generation_model}")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {str(e)}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query
            k: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vector_store:
            print("❌ Vector store not available")
            return []
        
        try:
            print(f"🔍 Searching for relevant chunks...")
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            relevant_chunks = []
            for i, (doc, score) in enumerate(results):
                chunk_info = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata,
                    'source_file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
                }
                relevant_chunks.append(chunk_info)
            
            print(f"📊 Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error during retrieval: {str(e)}")
            return []
    
    def generate_response(self, query: str, relevant_chunks: List[Dict], max_retries: int = 3) -> str:
        """
        Generate a response using Gemini 2.5 Pro based on query and relevant chunks.
        
        Args:
            query: User query
            relevant_chunks: List of relevant document chunks
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response
        """
        if not self.genai_client:
            return "❌ Generation model not available"
        
        if not relevant_chunks:
            return "❌ No relevant information found to answer your query."
        
        # Prepare context from relevant chunks
        context_parts = []
        for chunk in relevant_chunks:
            source_info = f"Source: {chunk['source_file']}, Page: {chunk['page']}"
            context_parts.append(f"[{source_info}]\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt for Gemini 2.5 Pro
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context. 

Use the following context to answer the user's question. If the answer is not clearly available in the context, say so.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, comprehensive answer based on the context
- Include relevant details and specifics from the documents
- If you reference specific information, mention which document/page it comes from
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Be concise but thorough

ANSWER:"""

        # Generate response with retry logic
        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating response with {self.generation_model}...")
                
                response = self.genai_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config=genai.GenerationConfig(
                        temperature=0.1,
                        top_p=0.8,
                        max_output_tokens=2000,
                    )
                )
                
                if response.candidates and len(response.candidates) > 0:
                    generated_text = response.candidates[0].content.parts[0].text
                    print(f"✅ Response generated successfully")
                    return generated_text
                else:
                    return "❌ No response generated"
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = 2 ** attempt
                    print(f"⏳ Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        return f"❌ Failed to generate response after {max_retries} attempts due to rate limiting"
                else:
                    print(f"❌ Error generating response: {str(e)}")
                    return f"❌ Error generating response: {str(e)}"
        
        return "❌ Failed to generate response"
    
    def query(self, question: str, k: int = 5) -> Dict:
        """
        Complete RAG query: retrieve relevant chunks and generate response.
        
        Args:
            question: User question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing query results
        """
        print("=" * 80)
        print(f"🔍 RAG QUERY: {question}")
        print("=" * 80)
        
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, k=k)
        
        if not relevant_chunks:
            return {
                'query': question,
                'answer': "❌ No relevant information found in the document.",
                'sources': [],
                'chunks_retrieved': 0
            }
        
        # Step 2: Generate response
        print(f"\n🤖 GENERATING RESPONSE...")
        answer = self.generate_response(question, relevant_chunks)
        
        # Prepare sources info
        sources = []
        for chunk in relevant_chunks:
            source_info = {
                'file': chunk['source_file'],
                'page': chunk['page'],
                'score': chunk['score']
            }
            if source_info not in sources:
                sources.append(source_info)
        
        result = {
            'query': question,
            'answer': answer,
            'sources': sources,
            'chunks_retrieved': len(relevant_chunks),
            'relevant_chunks': relevant_chunks
        }
        
        print(f"\n📝 FINAL ANSWER:")
        print(answer)
        
        print(f"\n📚 SOURCES:")
        for source in sources:
            print(f"- {source['file']}, Page: {source['page']} (Score: {source['score']:.4f})")
        
        print("=" * 80)
        return result
    
    def get_system_status(self) -> Dict:
        """Get the current status of the RAG system."""
        return {
            'vector_store_exists': self._vector_store_exists(),
            'vector_store_path': self.vector_store_path,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'generation_model': self.generation_model,
            'components_initialized': all([
                self.embeddings is not None,
                self.vector_store is not None,
                self.genai_client is not None
            ])
        }

def main():
    """Main function for testing the RAG system."""
    # Check if GOOGLE_API_KEY is set
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key in the .env file")
        return
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Test with sample queries
        sample_queries = [
            "What is this document about?",
            "What are the key financial metrics mentioned?",
            "Summarize the main points"
        ]
        
        print("\n🧪 Testing RAG System with sample queries...")
        for query in sample_queries:
            result = rag_system.query(query, k=3)
            print(f"\n{'='*20} NEXT QUERY {'='*20}")
            time.sleep(1)
        
        # Show system status
        status = rag_system.get_system_status()
        print("\n📊 SYSTEM STATUS:")
        for key, value in status.items():
            print(f"- {key}: {value}")
        
        return rag_system
        
    except Exception as e:
        print(f"❌ RAG System failed: {str(e)}")
        return None

if __name__ == "__main__":
    rag_system = main()