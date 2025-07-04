# RAG Main - Complete RAG system with automatic data processing
# This checks if vector store exists, if not calls data_processing.py, then proceeds with RAG

import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
import chromadb

# Load environment variables
load_dotenv()

# Import data processing pipeline
from data_processing import main as run_data_processing, load_existing_vector_store, Config

class GeminiEmbeddings:
    """Custom embeddings class that uses Google Gemini embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = None
        self._load_client()
    
    def _load_client(self):
        """Load the Gemini client."""
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"❌ Error initializing Gemini client: {str(e)}")
            raise
    
    def _embed_with_retry(self, text: str, max_retries: int = 6, base_delay: float = 1.0) -> List[float]:
        """Embed a single text with retry logic for rate limiting."""
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                return result.embeddings[0].values
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = base_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        print(f"❌ Max retries reached for text: {text[:50]}...")
                        return []
                else:
                    print(f"❌ Error embedding text: {str(e)}")
                    return []
        return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Gemini."""
        if not self.client:
            raise ValueError("Gemini client not loaded")
        return self._embed_with_retry(text)

class RAGSystem:
    """Complete RAG system with automatic data processing and retrieval/generation."""
    
    def __init__(self, 
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "pdf_documents",
                 embedding_model: str = "gemini-embedding-exp-03-07",
                 generation_model: str = "gemini-2.5-pro"):
        """Initialize the RAG system with automatic setup."""
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.genai_client = None
        
        self._setup_system()
    
    def _check_vector_store_exists(self) -> bool:
        """Check if vector store exists and has data."""
        try:
            if not os.path.exists(self.vector_store_path):
                return False
            
            # Try to load the vector store
            client = chromadb.PersistentClient(path=self.vector_store_path)
            collections = client.list_collections()
            
            # Check if our collection exists
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                collection = client.get_collection(self.collection_name)
                count = collection.count()
                print(f"📊 Found existing vector store with {count} documents")
                return count > 0
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking vector store: {str(e)}")
            return False
    
    def _setup_system(self):
        """Initialize RAG system - process data if needed, then setup components."""
        print("🚀 Initializing RAG System...")
        
        # Check if vector store exists
        if not self._check_vector_store_exists():
            print("📋 Vector store not found or empty. Running data processing pipeline...")
            
            # Run data processing pipeline
            documents, chunks, vector_store = run_data_processing()
            
            if not vector_store:
                print("❌ Data processing failed. Cannot initialize RAG system.")
                return
            
            print("✅ Data processing completed successfully!")
        else:
            print("✅ Vector store found. Loading existing data...")
        
        # Initialize embedding model
        try:
            self.embeddings = GeminiEmbeddings(self.embedding_model)
            print(f"✅ Embedding model loaded: {self.embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {str(e)}")
            return
        
        # Load vector store
        try:
            self.vector_store = load_existing_vector_store(
                persist_directory=self.vector_store_path,
                embedding_model_name=self.embedding_model,
                collection_name=self.collection_name
            )
            if not self.vector_store:
                print("❌ Failed to load vector store")
                return
        except Exception as e:
            print(f"❌ Failed to load vector store: {str(e)}")
            return
        
        # Initialize generation client
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.genai_client = genai.Client(api_key=api_key)
            print(f"✅ Generation model initialized: {self.generation_model}")
        except Exception as e:
            print(f"❌ Failed to initialize generation model: {str(e)}")
            return
        
        print("✅ RAG System ready!")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 8) -> List[Dict]:
        """Retrieve relevant document chunks for a query."""
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
        """Generate a response using Gemini 2.5 Pro based on query and relevant chunks."""
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
        
        # Create prompt for Gemini 2.5 Pro - Enhanced for numerical data and tables
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context. 

Use the following context to answer the user's question. Pay special attention to numerical data, percentages, and tabular information.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, comprehensive answer based on the context
- When dealing with numerical data or percentages, carefully examine ALL provided data
- If the question asks for "fastest growing" or "highest percentage", find the MAXIMUM value across all data
- Include relevant details and specifics from the documents
- If you reference specific information, mention which document/page it comes from
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Be precise with numerical values and percentages
- Double-check your answer against the provided data

ANSWER:"""

        # Generate response with retry logic
        for attempt in range(max_retries):
            try:
                print(f"🤖 Generating response with {self.generation_model}...")
                
                response = self.genai_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config={
                        'temperature': 0.1,
                        'top_p': 0.8,
                        'max_output_tokens': 2000,
                    }
                )
                
                # Extract the generated text
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
    
    def query(self, question: str, k: int = 8) -> Dict:
        """Complete RAG query: retrieve relevant chunks and generate response."""
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
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready to process queries."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None
        ])

def main():
    """Main function for testing the RAG system."""
    # Initialize RAG system
    rag_system = RAGSystem()
    
    if not rag_system.is_ready():
        print("❌ RAG system initialization failed!")
        return
    
    # Test with sample queries
    sample_queries = [
        "Which is the fastest growing demand booster?",
        "What are the key financial metrics mentioned?",
        "Summarize the executive summary",
        "What are the main business highlights?"
    ]
    
    print("\n🧪 Testing RAG System...")
    for query in sample_queries:
        result = rag_system.query(query, k=3)
        print(f"\n{'='*20} NEXT QUERY {'='*20}")
        time.sleep(1)  # Small delay between queries

if __name__ == "__main__":
    main()