# Robust RAG System with Intelligent Chunk Merging and Table Handling

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
import chromadb
import re
from collections import defaultdict

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

class SmartChunkMerger:
    """Intelligently merge overlapping and related chunks to reconstruct complete information."""
    
    @staticmethod
    def detect_table_content(text: str) -> bool:
        """Detect if content contains tabular data."""
        table_indicators = [
            r'\b\d+\s+\w+.*\d+%',  # Pattern like "1 Company Name ... 25%"
            r'Top\s+\d+\s+.*Accounts',  # "Top 20 ... Accounts"
            r'\d+\.\d+\s+\d+\.\d+',  # Multiple decimal numbers
            r'\b\d+%\s+\d+%',  # Multiple percentages
            r'Growth.*%.*\n.*\d+%',  # Growth percentage patterns
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def calculate_overlap_score(chunk1: str, chunk2: str) -> float:
        """Calculate content overlap between two chunks."""
        # Split into words for comparison
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @classmethod
    def group_related_chunks(cls, chunks: List[Dict]) -> List[List[Dict]]:
        """Group chunks that are related and should be merged."""
        if not chunks:
            return []
        
        # Group by page first
        page_groups = defaultdict(list)
        for chunk in chunks:
            page = chunk['metadata'].get('page', 'unknown')
            source = chunk['metadata'].get('source_file', 'unknown')
            key = f"{source}_{page}"
            page_groups[key].append(chunk)
        
        # Within each page, group by content similarity
        final_groups = []
        
        for page_key, page_chunks in page_groups.items():
            if len(page_chunks) == 1:
                final_groups.append(page_chunks)
                continue
            
            # Sort chunks by chunk_id to maintain order
            page_chunks.sort(key=lambda x: x['metadata'].get('chunk_id', 0))
            
            # Check if any chunks contain table content
            has_table = any(cls.detect_table_content(chunk['content']) for chunk in page_chunks)
            
            if has_table:
                # If page has table content, merge all chunks from this page
                final_groups.append(page_chunks)
            else:
                # For non-table content, group by overlap
                groups = []
                for chunk in page_chunks:
                    added_to_group = False
                    for group in groups:
                        # Check overlap with any chunk in the group
                        for group_chunk in group:
                            if cls.calculate_overlap_score(chunk['content'], group_chunk['content']) > 0.3:
                                group.append(chunk)
                                added_to_group = True
                                break
                        if added_to_group:
                            break
                    
                    if not added_to_group:
                        groups.append([chunk])
                
                final_groups.extend(groups)
        
        return final_groups
    
    @classmethod
    def merge_chunk_group(cls, chunk_group: List[Dict]) -> Dict:
        """Merge a group of related chunks into a single chunk."""
        if len(chunk_group) == 1:
            return chunk_group[0]
        
        # Sort by chunk_id to maintain logical order
        sorted_chunks = sorted(chunk_group, key=lambda x: x['metadata'].get('chunk_id', 0))
        
        # Combine content intelligently
        combined_content = ""
        seen_content = set()
        
        for chunk in sorted_chunks:
            content = chunk['content'].strip()
            
            # Avoid exact duplicates
            if content not in seen_content:
                combined_content += content + "\n\n"
                seen_content.add(content)
        
        # Create merged chunk with combined metadata
        merged_chunk = {
            'rank': sorted_chunks[0]['rank'],
            'content': combined_content.strip(),
            'score': min(chunk['score'] for chunk in sorted_chunks),  # Use best score
            'metadata': sorted_chunks[0]['metadata'].copy(),
            'source_file': sorted_chunks[0]['source_file'],
            'page': sorted_chunks[0]['page'],
            'chunk_id': f"merged_{sorted_chunks[0]['chunk_id']}_to_{sorted_chunks[-1]['chunk_id']}",
            'merged_from': len(sorted_chunks)
        }
        
        return merged_chunk

class RobustRAGSystem:
    """RAG system with intelligent chunk merging for complete table reconstruction."""
    
    def __init__(self, 
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "pdf_documents",
                 embedding_model: str = "gemini-embedding-exp-03-07",
                 generation_model: str = "gemini-2.5-pro"):
        """Initialize the robust RAG system."""
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.genai_client = None
        self.chunk_merger = SmartChunkMerger()
        
        self._setup_system()
    
    def _check_vector_store_exists(self) -> bool:
        """Check if vector store exists and has data."""
        try:
            if not os.path.exists(self.vector_store_path):
                return False
            
            client = chromadb.PersistentClient(path=self.vector_store_path)
            collections = client.list_collections()
            
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
        """Initialize RAG system with robust table handling."""
        print("🚀 Initializing Robust RAG System with Smart Chunk Merging...")
        
        # Check if vector store exists
        if not self._check_vector_store_exists():
            print("📋 Vector store not found or empty. Running data processing pipeline...")
            
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
        
        print("✅ Robust RAG System ready with intelligent chunk merging!")
    
    def retrieve_and_merge_chunks(self, query: str, k: int = 15) -> List[Dict]:
        """Retrieve chunks and intelligently merge related ones."""
        if not self.vector_store:
            print("❌ Vector store not available")
            return []
        
        try:
            print(f"🔍 Searching for relevant chunks (retrieving {k})...")
            
            # Retrieve more chunks to ensure we get complete information
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format initial results
            raw_chunks = []
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
                raw_chunks.append(chunk_info)
            
            print(f"📊 Retrieved {len(raw_chunks)} raw chunks")
            
            # Group and merge related chunks
            print("🔗 Analyzing chunk relationships and merging...")
            chunk_groups = self.chunk_merger.group_related_chunks(raw_chunks)
            
            merged_chunks = []
            for group in chunk_groups:
                merged_chunk = self.chunk_merger.merge_chunk_group(group)
                merged_chunks.append(merged_chunk)
            
            # Sort by relevance score
            merged_chunks.sort(key=lambda x: x['score'])
            
            print(f"✅ Merged into {len(merged_chunks)} intelligent chunks")
            
            # Log merging info
            for chunk in merged_chunks:
                if 'merged_from' in chunk:
                    print(f"   📋 Merged chunk from {chunk['merged_from']} original chunks (Page: {chunk['page']})")
            
            return merged_chunks
            
        except Exception as e:
            print(f"❌ Error during retrieval and merging: {str(e)}")
            return []
    
    def generate_response(self, query: str, merged_chunks: List[Dict], max_retries: int = 3) -> str:
        """Generate response using merged chunks."""
        if not self.genai_client:
            return "❌ Generation model not available"
        
        if not merged_chunks:
            return "❌ No relevant information found to answer your query."
        
        # Prepare context from merged chunks
        context_parts = []
        for chunk in merged_chunks:
            source_info = f"Source: {chunk['source_file']}, Page: {chunk['page']}"
            if 'merged_from' in chunk:
                source_info += f" (Merged from {chunk['merged_from']} chunks)"
            context_parts.append(f"[{source_info}]\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt for complete data analysis
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context. 

The context below contains COMPLETE and MERGED information from related document sections to ensure no data is missed.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Analyze ALL the provided data comprehensively
- When dealing with numerical data, percentages, or tables, examine EVERY entry
- If asking for "fastest growing", "highest", "maximum", or "best", find the ABSOLUTE maximum across ALL data
- The context has been intelligently merged to provide complete information - use all of it
- Include specific values, percentages, and source references
- Be precise with numerical values and double-check against ALL provided data
- If multiple similar data points exist, compare them all and identify the true maximum/minimum

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
    
    def query(self, question: str, k: int = 15) -> Dict:
        """Complete robust RAG query with intelligent chunk merging."""
        print("=" * 80)
        print(f"🔍 ROBUST RAG QUERY: {question}")
        print("=" * 80)
        
        # Step 1: Retrieve and merge chunks
        merged_chunks = self.retrieve_and_merge_chunks(question, k=k)
        
        if not merged_chunks:
            return {
                'query': question,
                'answer': "❌ No relevant information found in the document.",
                'sources': [],
                'chunks_retrieved': 0
            }
        
        # Step 2: Generate response
        print(f"\n🤖 GENERATING RESPONSE...")
        answer = self.generate_response(question, merged_chunks)
        
        # Prepare sources info
        sources = []
        for chunk in merged_chunks:
            source_info = {
                'file': chunk['source_file'],
                'page': chunk['page'],
                'score': chunk['score'],
                'merged_from': chunk.get('merged_from', 1)
            }
            if source_info not in sources:
                sources.append(source_info)
        
        result = {
            'query': question,
            'answer': answer,
            'sources': sources,
            'chunks_retrieved': len(merged_chunks),
            'merged_chunks': merged_chunks
        }
        
        print(f"\n📝 FINAL ANSWER:")
        print(answer)
        
        print(f"\n📚 SOURCES (Merged):")
        for source in sources:
            merge_info = f" (Merged from {source['merged_from']} chunks)" if source['merged_from'] > 1 else ""
            print(f"- {source['file']}, Page: {source['page']} (Score: {source['score']:.4f}){merge_info}")
        
        print("=" * 80)
        return result
    
    def is_ready(self) -> bool:
        """Check if the robust RAG system is ready."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None
        ])

# Alias for backward compatibility
RAGSystem = RobustRAGSystem

def main():
    """Test the robust RAG system."""
    rag_system = RobustRAGSystem()
    
    if not rag_system.is_ready():
        print("❌ Robust RAG system initialization failed!")
        return
    
    # Test queries
    test_queries = [
        "Which is the fastest growing demand booster based on growth %?",
        "What are all the growth percentages in the top 20 demand booster accounts?",
        "List all companies with growth over 100%"
    ]
    
    print("\n🧪 Testing Robust RAG System...")
    for query in test_queries:
        result = rag_system.query(query, k=15)
        print(f"\n{'='*20} NEXT QUERY {'='*20}")
        time.sleep(1)

if __name__ == "__main__":
    main()