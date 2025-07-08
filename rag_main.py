# Enhanced RAG System with CFA Agent and Self-Query Mechanism

import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
import chromadb
from collections import defaultdict
from datetime import datetime
from enum import Enum
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Import data processing pipeline
from data_processing import main as run_data_processing, load_existing_vector_store, Config

class QueryType(Enum):
    DIRECT_FACTUAL = "direct_factual"
    ANALYTICAL = "analytical"

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
            logging.error(f"❌ Error initializing Gemini client: {str(e)}")
            raise
    
    def _embed_with_retry(self, text: str, max_retries: int = 6, base_delay: float = 1.0) -> List[float]:
        """Embed a single text with retry logic for rate limiting."""
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                # Defensive check for result structure
                if result and hasattr(result, 'embeddings') and result.embeddings and \
                   hasattr(result.embeddings[0], 'values') and result.embeddings[0].values is not None:
                    return list(result.embeddings[0].values)
                else:
                    logging.error(f"❌ Unexpected embedding response: {result}")
                    return []
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"⏳ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        logging.error(f"❌ Max retries reached for text: {text[:50]}...")
                        return []
                else:
                    logging.error(f"❌ Error embedding text: {str(e)}")
                    return []
        return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Gemini."""
        if not self.client:
            raise ValueError("Gemini client not loaded")
        return self._embed_with_retry(text)

class QueryClassifier:
    """Classifies queries as direct factual or analytical requiring CFA agent."""
    
    @staticmethod
    def classify_query(query: str) -> QueryType:
        """Classify the query type based on intent."""
        analytical_keywords = [
            'compare', 'analyze', 'analyse', 'why', 'reason', 'cause', 'trend', 'growth', 'decline',
            'increase', 'decrease', 'performance', 'vs', 'versus', 'difference', 'impact',
            'correlation', 'relationship', 'factor', 'driver', 'explain', 'understand'
        ]
        
        query_lower = query.lower()
        
        # Check for analytical keywords
        if any(keyword in query_lower for keyword in analytical_keywords):
            return QueryType.ANALYTICAL
        
        # Check for multiple segments/products mentioned (likely comparative)
        segments = ['daas', 'distribution', 'martech']
        products = ['travel bi', 'hospi bi', 'enterprise connectivity', 'channel manager', 
                   'uno', 'bcv', 'mhs', 'adara']
        
        mentioned_count = sum(1 for item in segments + products if item in query_lower)
        if mentioned_count > 1:
            return QueryType.ANALYTICAL
        
        return QueryType.DIRECT_FACTUAL

class SubQueryGenerator:
    """Generates sub-queries for CFA deep analysis."""
    
    def __init__(self, genai_client):
        self.genai_client = genai_client
    
    def generate_sub_queries(self, original_query: str, context: str = "") -> List[str]:
        """Generate sub-queries for deep financial analysis."""
        
        prompt = f"""You are a Chartered Financial Analyst. Given the user's analytical query about RateGain financial data, generate a list of specific sub-queries that need to be answered to provide a comprehensive analysis.

BUSINESS CONTEXT:
- RateGain has 3 segments: DaaS (Travel BI, Hospi BI), Distribution (Enterprise Connectivity, Channel Manager, Uno), Martech (BCV, MHS, Adara)
- Available data: Revenue, EBITDA, Costs, Top Accounts, NRR, GRR, Monetization, Department Spending
- Time period: April 2024 - March 2025

CONVERSATION CONTEXT:
{context}

USER QUERY: {original_query}

Generate 5-8 specific sub-queries that will help analyze this comprehensively. Include queries about:
1. Base metrics (EBITDA, Revenue for specific periods)
2. Supporting data (Top accounts, costs, department spending)
3. Comparative analysis if multiple periods/products mentioned

Return only the sub-queries, one per line, without numbering or explanations."""

        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={'temperature': 0.3, 'max_output_tokens': 5000}
            )
            
            if response and response.candidates and len(response.candidates) > 0:
                sub_queries_text = response.candidates[0].content.parts[0].text
                sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
                return sub_queries[:8]  # Limit to 8 sub-queries
            
        except Exception as e:
            logging.error(f"Error generating sub-queries: {e}")
        
            return []

class CFAAgent:
    """Chartered Financial Analyst agent for deep financial analysis."""
    
    def __init__(self, vector_store, embeddings, genai_client):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.genai_client = genai_client
        self.sub_query_generator = SubQueryGenerator(genai_client)
    
    def analyze_with_thinking(self, query: str, context: str = "") -> Generator[Dict, None, None]:
        """Perform deep financial analysis with live thinking display."""
        
        logging.info("🧠 **THINKING**: Starting CFA analysis...")
        
        # Generate sub-queries
        logging.info("🔍 **THINKING**: Generating analytical sub-queries...")
        sub_queries = self.sub_query_generator.generate_sub_queries(query, context)
        
        if not sub_queries:
            logging.warning("⚠️ **THINKING**: Using fallback analysis approach...")
            sub_queries = [query]  # Fallback to original query
        
        # Display generated sub-queries
        sub_queries_text = "\n".join([f"• {q}" for q in sub_queries])
        logging.info(f"📋 **THINKING**: Generated {len(sub_queries)} sub-queries:\n{sub_queries_text}")
        
        # Collect all data
        all_retrieved_data = []
        
        for i, sub_query in enumerate(sub_queries, 1):
            logging.info(f"🔍 **THINKING**: Processing sub-query {i}/{len(sub_queries)}: {sub_query}")
            
            # Retrieve relevant chunks
            try:
                results = self.vector_store.similarity_search_with_score(sub_query, k=7)
                
                for doc, score in results:
                    chunk_data = {
                        'content': doc.page_content,
                        'score': score,
                        'metadata': doc.metadata,
                        'sub_query': sub_query
                    }
                    all_retrieved_data.append(chunk_data)
                
                logging.info(f"✅ **THINKING**: Retrieved {len(results)} chunks for sub-query {i}")
                
            except Exception as e:
                logging.error(f"❌ **THINKING**: Error retrieving data for sub-query {i}: {str(e)}")
        
        # Remove duplicates and sort by relevance
        unique_data = []
        seen_content = set()
        for data in all_retrieved_data:
            if data['content'] not in seen_content:
                unique_data.append(data)
                seen_content.add(data['content'])
        
        unique_data.sort(key=lambda x: x['score'])
        
        logging.info(f"📊 **THINKING**: Compiled {len(unique_data)} unique chunks for analysis")
        
        # Generate comprehensive analysis
        logging.info("🤖 **THINKING**: Performing comprehensive financial analysis...")
        
        analysis = self._generate_cfa_analysis(query, unique_data, context)
        
        logging.info("✅ **THINKING**: Analysis complete!")
        yield {"type": "answer", "content": analysis, "sources": self._format_sources(unique_data)}
    
    def _generate_cfa_analysis(self, query: str, retrieved_data: List[Dict], context: str) -> str:
        """Generate comprehensive CFA analysis."""
        
        # Prepare context from retrieved data
        context_parts = []
        for data in retrieved_data:
            source_info = f"Source: {data['metadata'].get('source_file', 'Unknown')}, Page: {data['metadata'].get('page', 'Unknown')}"
            context_parts.append(f"[{source_info}]\n{data['content']}\n")
        
        full_context = "\n".join(context_parts)
        
        prompt = f"""You are a senior Chartered Financial Analyst (CFA) specializing in RateGain's financial performance. Provide a comprehensive financial analysis based on the data provided.

BUSINESS STRUCTURE:
- DaaS Segment: Travel BI, Hospi BI  
- Distribution Segment: Enterprise Connectivity, Channel Manager, Uno
- Martech Segment: BCV, MHS, Adara

CONVERSATION CONTEXT:
{context}

FINANCIAL DATA:
{full_context}

USER QUERY: {query}

ANALYSIS REQUIREMENTS:
1. **Executive Summary**: Start with key findings
2. **Detailed Financial Analysis**: 
   - Analyze EBITDA, Revenue, Costs systematically
   - Identify trends, variances, and performance drivers
   - Examine top accounts and customer dynamics
   - Review department spending patterns
3. **Root Cause Analysis**: Explain the "why" behind numbers
4. **Business Implications**: What this means for RateGain
5. **Data-Driven Insights**: Include specific numbers, percentages, and comparisons

IMPORTANT:
- Be factually accurate with all numbers
- Reference specific time periods correctly (FY 2024-25: Apr 2024 - Mar 2025)
- Provide actionable business insights
- Use professional financial analysis language
- Include specific account names and financial figures when available

ANALYSIS:"""

        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'max_output_tokens': 10000,
                }
            )
            # Defensive check for response structure
            if response and hasattr(response, "candidates") and response.candidates and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts and \
               response.candidates[0].content.parts[0] is not None and \
               hasattr(response.candidates[0].content.parts[0], 'text') and response.candidates[0].content.parts[0].text is not None:
                return response.candidates[0].content.parts[0].text
            else:
                logging.error(f"❌ Unexpected CFA analysis response: {response}")
                return "❌ Unable to generate analysis"
                
        except Exception as e:
            logging.error(f"❌ Error generating analysis: {str(e)}")
            return "❌ Error generating analysis"
    
    def _format_sources(self, retrieved_data: List[Dict]) -> List[Dict]:
        """Format sources for display."""
        sources = []
        seen_sources = set()
        
        for data in retrieved_data:
            source_key = f"{data['metadata'].get('source_file', 'Unknown')}_{data['metadata'].get('page', 'Unknown')}"
            if source_key not in seen_sources:
                sources.append({
                    'file': data['metadata'].get('source_file', 'Unknown'),
                    'page': data['metadata'].get('page', 'Unknown'),
                    'score': data['score']
                })
                seen_sources.add(source_key)
        
        return sorted(sources, key=lambda x: x['score'])[:10]  # Top 10 sources

class EnhancedRAGSystem:
    """Enhanced RAG system with CFA agent and intelligent query routing."""
    
    def __init__(self, 
                 vector_store_path: str = "vector_store",
                 collection_name: str = "pdf_documents",
                 embedding_model: str = "gemini-embedding-exp-03-07",
                 generation_model: str = "gemini-2.5-pro"):
        """Initialize the enhanced RAG system."""
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.genai_client = None
        self.cfa_agent = None
        self.query_classifier = QueryClassifier()
        
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
                logging.info(f"📊 Found existing vector store with {count} documents")
                return count > 0
            
            return False
            
        except Exception as e:
            logging.warning(f"⚠️ Error checking vector store: {str(e)}")
            return False
    
    def _setup_system(self):
        """Initialize enhanced RAG system."""
        logging.info("🚀 Initializing Enhanced RAG System with CFA Agent...")
        
        # Check if vector store exists
        if not self._check_vector_store_exists():
            logging.info("📋 Vector store not found or empty. Running data processing pipeline...")
            documents, chunks, vector_store = run_data_processing()
            if not vector_store:
                logging.error("❌ Data processing failed. Cannot initialize RAG system.")
                return
            logging.info("✅ Data processing completed successfully!")
        else:
            logging.info("✅ Vector store found. Loading existing data...")
        
        # Initialize embedding model
        try:
            self.embeddings = GeminiEmbeddings(self.embedding_model)
            logging.info(f"✅ Embedding model loaded: {self.embedding_model}")
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model: {str(e)}")
            return
        
        # Load vector store
        try:
            self.vector_store = load_existing_vector_store(
                persist_directory=self.vector_store_path,
                embedding_model_name=self.embedding_model,
                collection_name=self.collection_name
            )
            if not self.vector_store:
                logging.error("❌ Failed to load vector store")
                return
        except Exception as e:
            logging.error(f"❌ Failed to load vector store: {str(e)}")
            return
        
        # Initialize generation client
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.genai_client = genai.Client(api_key=api_key)
            logging.info(f"✅ Generation model initialized: {self.generation_model}")
        except Exception as e:
            logging.error(f"❌ Failed to initialize generation model: {str(e)}")
            return
        
        # Initialize CFA agent
        self.cfa_agent = CFAAgent(self.vector_store, self.embeddings, self.genai_client)
        logging.info("✅ CFA Agent initialized")
        
        logging.info("✅ Enhanced RAG System ready!")
    
    def query(self, question: str, context: str = "") -> Generator[Dict, None, None]:
        """Process query with intelligent routing."""
        if not self.is_ready():
            yield {"type": "error", "content": "❌ RAG system not ready"}
            return
        
        # Classify query type
        query_type = self.query_classifier.classify_query(question)
        logging.info(f"📋 **PROCESSING**: Query type classified as {query_type.name}")
        
        if query_type == QueryType.DIRECT_FACTUAL:
            # Handle direct factual queries
            logging.info("📋 **PROCESSING**: Direct factual query detected")
            
            try:
                results = self.vector_store.similarity_search_with_score(question, k=10)
                logging.debug(f"✅ Direct factual query: Retrieved {len(results)} chunks")
                
                # Prepare context
                context_parts = []
                sources = []
                for doc, score in results:
                    source_info = f"Source: {doc.metadata.get('source_file', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}"
                    context_parts.append(f"[{source_info}]\n{doc.page_content}\n")
                    sources.append({
                        'file': doc.metadata.get('source_file', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'score': score
                    })
                
                full_context = "\n".join(context_parts)
                
                # Generate direct answer
                prompt = f"""You are a financial analyst assistant. Answer the user's question directly based on the provided RateGain financial data.\n\nCONVERSATION CONTEXT:\n{context}\n\nFINANCIAL DATA:\n{full_context}\n\nUSER QUESTION: {question}\n\nProvide a direct, accurate answer with specific numbers and source references. Be concise but complete."""
                response = self.genai_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config={'temperature': 0.1, 'max_output_tokens': 5000}
                )
                # Defensive check for response structure
                if response and hasattr(response, "candidates") and response.candidates and \
                   hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
                   hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts and \
                   response.candidates[0].content.parts[0] is not None and \
                   hasattr(response.candidates[0].content.parts[0], 'text') and response.candidates[0].content.parts[0].text is not None:
                    answer = response.candidates[0].content.parts[0].text
                    logging.info("✅ Direct factual query: Answer generated successfully")
                    yield {"type": "answer", "content": answer, "sources": sources[:5]}
                else:
                    logging.error(f"❌ Direct factual query: Unexpected response: {response}")
                    yield {"type": "error", "content": "❌ Unable to generate response"}
                
            except Exception as e:
                logging.error(f"❌ Error processing direct factual query: {str(e)}")
                yield {"type": "error", "content": f"❌ Error processing query: {str(e)}"}
        
        else:  # ANALYTICAL query
            logging.info("🧠 **PROCESSING**: Analytical query detected - routing to CFA Agent")
            
            # Use CFA agent for deep analysis
            yield from self.cfa_agent.analyze_with_thinking(question, context)
    
    def is_ready(self) -> bool:
        """Check if the enhanced RAG system is ready."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None,
            self.cfa_agent is not None
        ])

# Alias for compatibility
RAGSystem = EnhancedRAGSystem

def main():
    """Test the enhanced RAG system."""
    rag_system = EnhancedRAGSystem()
    
    if not rag_system.is_ready():
        logging.error("❌ Enhanced RAG system initialization failed!")
        return
    
    # Test queries
    test_queries = [
        "What was the GAAP revenue for Hospi BI in August 2024?",  # Direct
        "Compare the EBITDA for Hospi BI and Travel BI in Q2 and Q3, analyze why there has been any increase or decrease"  # Analytical
    ]
    
    logging.info("\n🧪 Testing Enhanced RAG System...")
    for query in test_queries:
        logging.info(f"\n{'='*50}")
        logging.info(f"QUERY: {query}")
        logging.info('='*50)
        
        for response in rag_system.query(query):
            if response["type"] == "thinking":
                logging.info(response["content"])
            elif response["type"] == "answer":
                logging.info(f"\n📝 FINAL ANSWER:")
                logging.info(response["content"])
                if response.get("sources"):
                    logging.info(f"\n📚 SOURCES:")
                    for source in response["sources"]:
                        logging.info(f"- {source['file']}, Page: {source['page']} (Score: {source['score']:.4f})")
            elif response["type"] == "error":
                logging.error(response["content"])
        
        time.sleep(2)

if __name__ == "__main__":
    main()