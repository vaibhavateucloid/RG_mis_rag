# Enhanced RAG System with March PDF Prioritization - Optimized Retrieval

import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
import chromadb
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
import logging
import typing

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Import data processing pipeline
from image_embedding import main as run_data_processing, load_existing_vector_store, Config

class QueryType(Enum):
    DIRECT_FACTUAL = "direct_factual"
    EXECUTIVE_ANALYTICAL = "executive_analytical"
    DEEP_ANALYTICAL = "deep_analytical"

class MarchPDFRetrieval:
    """Smart retrieval system that prioritizes March PDF for company-wide/FY queries."""
    
    @staticmethod
    def is_company_wide_query(query: str) -> bool:
        """Detect if query is asking for company-wide, quarterly, or FY data."""
        query_lower = query.lower()
        
        # Company-wide indicators
        company_indicators = [
            'full company', 'entire company', 'overall company', 'company wide', 'company-wide',
            'rategain', 'consolidated', 'overall', 'total', 'combined', 'aggregate',
            'company as a whole', 'whole company', 'entire business', 'full business'
        ]
        
        # Fiscal year indicators
        fy_indicators = [
            'fy 24-25', 'fy24-25', 'fy 2024-25', 'fy 2025', 'fiscal year 24-25',
            'fiscal year 2024-25', 'fiscal year 2025', 'annual', 'yearly',
            'full year', 'entire year', 'april 2024 to march 2025',
            'april 2024 - march 2025', '12 months', 'full fiscal'
        ]
        
        # Quarterly indicators
        quarterly_indicators = [
            'q1', 'q2', 'q3', 'q4', 'quarter', 'quarterly', 'q1 fy25', 'q2 fy25',
            'q3 fy25', 'q4 fy25', 'first quarter', 'second quarter', 'third quarter',
            'fourth quarter', 'quarter ended', 'quarterly results'
        ]
        
        # Check for any indicators
        all_indicators = company_indicators + fy_indicators + quarterly_indicators
        return any(indicator in query_lower for indicator in all_indicators)
    
    @staticmethod
    def prioritized_retrieval(vector_store, query: str, k: int = 15) -> List[Tuple]:
        """
        Hybrid retrieval that prioritizes March PDF for company-wide queries.
        Returns list of (document, score) tuples.
        """
        results = []
        
        if MarchPDFRetrieval.is_company_wide_query(query):
            logging.info(f"🎯 Company-wide query detected, prioritizing March PDF")
            
            # Step 1: Get March PDF chunks specifically
            march_results = MarchPDFRetrieval._retrieve_from_march_pdf(vector_store, query, k=8)
            results.extend(march_results)
            logging.info(f"📊 Retrieved {len(march_results)} chunks from March PDF")
            
            # Step 2: Supplement with general search, avoiding duplicates
            general_results = vector_store.similarity_search_with_score(query, k=k)
            march_content = {doc.page_content for doc, _ in march_results}
            
            for doc, score in general_results:
                if doc.page_content not in march_content and len(results) < k:
                    results.append((doc, score))
            
            logging.info(f"📈 Total chunks after supplementing: {len(results)}")
            
        else:
            # Regular semantic search for non-company-wide queries
            results = vector_store.similarity_search_with_score(query, k=k)
            logging.info(f"🔍 Regular semantic search: {len(results)} chunks")
        
        return results
    
    @staticmethod
    def _retrieve_from_march_pdf(vector_store, query: str, k: int = 8) -> List[Tuple]:
        """Retrieve chunks specifically from March PDF."""
        try:
            # Get all documents first with a larger k to find March PDF content
            all_results = vector_store.similarity_search_with_score(query, k=50)
            
            # Filter for March PDF documents
            march_results = []
            for doc, score in all_results:
                source_file = doc.metadata.get('source_file', '').lower()
                if 'march' in source_file or 'mar' in source_file:
                    march_results.append((doc, score))
                    if len(march_results) >= k:
                        break
            
            # If no March-specific files found, look for recent MIS files
            if not march_results:
                for doc, score in all_results:
                    source_file = doc.metadata.get('source_file', '').lower()
                    if 'mis' in source_file and any(month in source_file for month in ['march', 'mar', '03']):
                        march_results.append((doc, score))
                        if len(march_results) >= k:
                            break
            
            return march_results
            
        except Exception as e:
            logging.error(f"Error retrieving from March PDF: {e}")
            return []

class IntelligentRetrieval:
    """Enhanced intelligent retrieval with March PDF prioritization."""
    
    @staticmethod
    def classify_subquery_complexity(sub_query: str) -> str:
        """Classify sub-query complexity for optimal retrieval count."""
        query_lower = sub_query.lower()
        
        # Complex multi-dimensional queries
        complex_keywords = ['root cause', 'analyze', 'comprehensive', 'detailed analysis', 'drivers', 'trends', 'compare multiple', 'breakdown']
        if any(keyword in query_lower for keyword in complex_keywords):
            return 'complex'
        
        # Comparative queries
        comparative_keywords = ['compare', 'vs', 'versus', 'difference', 'contrast', 'relative']
        if any(keyword in query_lower for keyword in comparative_keywords):
            return 'comparative'
        
        # Simple factual queries
        simple_keywords = ['what was', 'what is', 'how much', 'show', 'list', 'value of', 'amount']
        if any(keyword in query_lower for keyword in simple_keywords):
            return 'simple'
        
        # Default to moderate complexity
        return 'moderate'
    
    @staticmethod
    def get_optimal_retrieval_count(sub_query: str, query_type: str = 'moderate') -> int:
        """Get optimal initial retrieval count based on query characteristics."""
        complexity = IntelligentRetrieval.classify_subquery_complexity(sub_query)
        
        # Increase retrieval counts for better coverage
        retrieval_map = {
            'simple': 8,
            'moderate': 12,
            'comparative': 15,
            'complex': 18
        }
        
        return retrieval_map.get(complexity, 12)
    
    @staticmethod
    def apply_relevance_filtering(results: List[Tuple], min_score: float = 0.2, score_drop_threshold: float = 0.2) -> List[Tuple]:
        """Apply intelligent relevance filtering with relaxed thresholds."""
        if not results:
            return results
        
        filtered_results = []
        prev_score = None
        
        for doc, score in results:
            # Always include first few results if they meet minimum threshold
            if len(filtered_results) < 5 and score >= min_score:
                filtered_results.append((doc, score))
                prev_score = score
                continue
            
            # Stop if score drops significantly
            if prev_score and (prev_score - score) > score_drop_threshold:
                break
            
            # Stop if score is too low
            if score < min_score:
                break
            
            filtered_results.append((doc, score))
            prev_score = score
            
            # Cap at reasonable maximum
            if len(filtered_results) >= 20:
                break
        
        # Ensure minimum results for complex queries
        if len(filtered_results) < 5 and len(results) >= 5:
            return results[:5]
        
        return filtered_results
    
    @staticmethod
    def enhanced_retrieve(vector_store, query: str, max_chunks: int = 20) -> List[Tuple]:
        """Enhanced retrieval using March PDF prioritization."""
        # Get optimal initial count
        optimal_count = IntelligentRetrieval.get_optimal_retrieval_count(query)
        initial_count = min(optimal_count, max_chunks)
        
        # Use prioritized retrieval
        results = MarchPDFRetrieval.prioritized_retrieval(vector_store, query, k=initial_count)
        
        # Apply relaxed relevance filtering
        filtered_results = IntelligentRetrieval.apply_relevance_filtering(results)
        
        logging.info(f"Enhanced retrieval: {len(results)} → {len(filtered_results)} chunks for query: {query[:50]}...")
        return filtered_results

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
    """Classifies queries as direct factual, executive analytical, or deep analytical."""
    
    def __init__(self, genai_client=None):
        self.conversation_history = deque(maxlen=5)
        self.genai_client = genai_client
    
    def add_to_history(self, query: str, response_type: str):
        """Add query and response type to conversation history."""
        self.conversation_history.append({
            'query': query,
            'response_type': response_type,
            'timestamp': datetime.now()
        })
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the query type using Gemini-2.5-Flash if available, else fallback to robust keyword logic."""
        query_lower = query.lower()
        
        # Try LLM-based classification if client is available
        if self.genai_client is not None:
            try:
                prompt = f"""
You are an expert assistant. Classify the following user query into one of three categories:
- direct_factual: The user is asking for specific numbers, facts, or metrics (e.g., 'What was the revenue in Q2?').
- executive_analytical: The user wants a high-level summary, overview, comparison, or business insight (e.g., 'Summarize the performance of Hospi BI in Q2', 'Overview of...', 'Compare X and Y').
- deep_analytical: The user wants a detailed, root-cause, or multi-step analysis (e.g., 'Analyze the EBITDA trends and explain the drivers', 'Why did revenue drop?').

User query: {query}

Respond with only one of: direct_factual, executive_analytical, deep_analytical. Do not explain.
"""
                response = self.genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        'temperature': 0.0,
                        'max_output_tokens': 5000
                    }
                )
                if response and hasattr(response, "candidates") and response.candidates and \
                   response.candidates[0] is not None and \
                   hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
                   hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                    label = response.candidates[0].content.parts[0].text.strip().lower()
                    if label == "direct_factual":
                        return QueryType.DIRECT_FACTUAL
                    elif label == "executive_analytical":
                        return QueryType.EXECUTIVE_ANALYTICAL
                    elif label == "deep_analytical":
                        return QueryType.DEEP_ANALYTICAL
            except Exception as e:
                logging.warning(f"[QueryClassifier] LLM classification failed, falling back to keyword logic: {e}")
        
        # Fallback: robust keyword-based logic
        deep_analysis_keywords = [
            'root cause', 'why', 'explain', 'driver', 'reason', 'cause', 'deep analysis', 'detailed analysis',
            'elaborate', 'breakdown', 'comprehensive analysis', 'in-depth', 'thorough', 'analyze', 'analyse',
            'further analysis', 'drill down', 'expand on', 'dive deeper', 'what caused', 'explain the reason', 'drivers'
        ]
        if any(kw in query_lower for kw in deep_analysis_keywords):
            return QueryType.DEEP_ANALYTICAL
        
        executive_keywords = [
            'summary', 'summarize', 'overview', 'insight', 'compare', 'comparison', 'trend', 'performance',
            'high level', 'business impact', 'implication', 'implications', 'key findings', 'key insights',
            'business insight', 'business overview', 'business summary', 'business review', 'review', 'synthesis',
            'recap', 'conclusion', 'conclude', 'highlight', 'main points', 'main findings', 'main insights', 'synthesize'
        ]
        if any(kw in query_lower for kw in executive_keywords):
            return QueryType.EXECUTIVE_ANALYTICAL
        
        metric_keywords = [
            'revenue', 'ebitda', 'cost', 'costs', 'profit', 'loss', 'nrr', 'grr', 'retention', 'monetization',
            'department spending', 'sales', 'ltv', 'cogs', 'cashflow', 'cash flow', 'collection', 'accounts',
            'account', 'customer', 'customers', 'number of', 'amount', 'total', 'value', 'figure', 'metric', 'data',
            'percentage', 'percent', 'ratio', 'score', 'count', 'average', 'mean', 'median', 'variance', 'change',
            'increase', 'decrease', 'drop', 'rise', 'growth', 'decline', 'month', 'quarter', 'year', 'period', 'date',
            'as of', 'for', 'in', 'during', 'between', 'show', 'list', 'give', 'provide', 'display', 'report', 'state',
            'what is', 'what was', 'how many', 'how much', 'when', 'which', 'find', 'identify', 'fetch', 'extract'
        ]
        
        if any(kw in query_lower for kw in metric_keywords) and not any(kw in query_lower for kw in executive_keywords + deep_analysis_keywords):
            return QueryType.DIRECT_FACTUAL
        
        return QueryType.EXECUTIVE_ANALYTICAL

class ExecutiveAgent:
    """Executive-level analytical agent for concise, comprehensive insights."""
    
    def __init__(self, vector_store, embeddings, genai_client, llm_fallback_fn):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.genai_client = genai_client
        self._call_llm_with_fallback = llm_fallback_fn
    
    def _intelligent_retrieve(self, query: str, max_chunks: int = 20) -> List[Tuple]:
        """Use enhanced retrieval with March PDF prioritization."""
        return IntelligentRetrieval.enhanced_retrieve(self.vector_store, query, max_chunks)
    
    async def analyze_executive_summary(self, query: str, context: str = "", original_query: str = None, chat_history: list = None) -> typing.AsyncGenerator[Dict, None]:
        """Perform executive-level analysis with concise insights."""
        import asyncio
        
        try:
            enhanced_query = query
            if chat_history and hasattr(self, '_parent_system'):
                enhanced_query = self._parent_system._contextualize_query(original_query or query, chat_history)
            
            # Use enhanced retrieval
            results = self._intelligent_retrieve(enhanced_query)
            
            # Prepare context from retrieved data
            context_parts = []
            sources = []
            for doc, score in results:
                context_parts.append(doc.page_content)
                sources.append({
                    'file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'score': score
                })
            
            full_context = "\n".join(context_parts)
            
            # Generate executive summary
            analysis = self._generate_executive_analysis(original_query or query, full_context, context, chat_history)
            
            yield {"type": "answer", "content": analysis, "sources": sources[:5]}
            
        except Exception as e:
            logging.error(f"❌ Error in executive analysis: {str(e)}")
            yield {"type": "error", "content": f"❌ Error generating executive summary: {str(e)}"}
    
    def _generate_executive_analysis(self, query: str, retrieved_context: str, conversation_context: str, chat_history: list) -> str:
        """Generate executive-level analysis with concise insights."""
        
        prompt = f"""You are a senior financial advisor for RateGain Travel Technologies. Provide a concise executive summary focused on actionable business insights.

COMPANY CONTEXT:
RateGain is a leading SaaS provider for travel/hospitality with 7000+ global customers across three segments:
• DaaS: Travel BI and Hospi BI (business intelligence)
• Distribution: Enterprise Connectivity, Channel Manager, Uno platform  
• Martech: BCV competitive intelligence, MHS marketing automation, Adara data platform

CONVERSATION CONTEXT: {conversation_context}

FINANCIAL DATA: {retrieved_context}

USER QUERY: {query}

EXECUTIVE SUMMARY REQUIREMENTS:
• Length: 200-300 words maximum
• Focus on business impact and strategic implications
• Include specific metrics and percentages from the data
• Highlight non-obvious insights or patterns
• Use bullet points for key findings
• Tables for comparative data when relevant
• No source citations in text

FORMAT:
## Executive Summary
[2-3 sentence overview with key takeaway]

## Key Insights
• [Insight 1 with specific data]
• [Insight 2 with trend/pattern]
• [Insight 3 with business impact]
• [Non-obvious finding]

## Strategic Implications
[Brief actionable recommendations]

---
💡 *For detailed analysis, ask me to "elaborate" or "analyze further"*

ANALYSIS:"""

        try:
            response = self._call_llm_with_fallback(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'max_output_tokens': 25000,
                }
            )
            if response and hasattr(response, "candidates") and response.candidates and \
               response.candidates[0] is not None and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                answer = ''.join([p.text for p in parts if hasattr(p, 'text') and p.text])
                if answer.strip():
                    return answer
                else:
                    logging.warning(f"LLM returned no answer text. Full response: {response}")
                    return "❌ No answer could be generated for this query."
            else:
                logging.error(f"❌ Unexpected executive analysis response: {response}")
                return "❌ Unable to generate executive summary"
        except Exception as e:
            logging.error(f"❌ Error generating executive analysis: {str(e)}")
            return "❌ Error generating executive summary"

class SubQueryGenerator:
    """Generates sub-queries for CFA deep analysis."""
    
    def __init__(self, genai_client, llm_fallback_fn):
        self.genai_client = genai_client
        self._call_llm_with_fallback = llm_fallback_fn
    
    def generate_sub_queries(self, original_query: str, context: str = "") -> List[str]:
        """Generate sub-queries for deep financial analysis."""
        prompt = f"""You are a Chartered Financial Analyst. Generate specific, focused sub-queries for comprehensive RateGain financial analysis.

COMPANY CONTEXT:
RateGain: Leading travel tech SaaS with 7000+ customers
• DaaS: Travel BI, Hospi BI  
• Distribution: Enterprise Connectivity, Channel Manager, Uno
• Martech: BCV, MHS, Adara

AVAILABLE METRICS: Revenue, EBITDA, Costs, NRR/GRR, Top Accounts, Department Spending, Rule of 40, LTV/CAC, Cashflow, Collections, Monetization
TIME PERIOD: April 2024 - March 2025

CONVERSATION CONTEXT: {context}
USER QUERY: {original_query}

Generate 6-8 specific sub-queries that directly support analyzing this query. Focus on:
• Base financial metrics relevant to the query
• Comparative analysis if multiple periods/segments mentioned  
• Key performance indicators (NRR, GRR, retention, top accounts)
• Operational metrics (department spending, monetization, cashflow)
• Only include sub-queries directly relevant to the user's question

Return sub-queries only, one per line, no numbering."""
        
        try:
            response = self._call_llm_with_fallback(
                model="gemini-2.5-flash",
                contents=prompt,
                config={'temperature': 0.4, 'max_output_tokens': 15000}
            )
            if response and hasattr(response, "candidates") and response.candidates and \
               response.candidates[0] is not None and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                sub_queries_text = response.candidates[0].content.parts[0].text
                sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
                return sub_queries[:8]
            return []
        except Exception as e:
            logging.error(f"Error generating sub-queries: {e}")
            return []

class CFAAgent:
    """Chartered Financial Analyst agent for deep financial analysis."""
    
    def __init__(self, vector_store, embeddings, genai_client, llm_fallback_fn):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.genai_client = genai_client
        self._call_llm_with_fallback = llm_fallback_fn
        self.sub_query_generator = SubQueryGenerator(genai_client, llm_fallback_fn)
    
    def _intelligent_retrieve(self, sub_query: str) -> List[Dict]:
        """Use enhanced retrieval for a sub-query."""
        # Use enhanced retrieval
        results = IntelligentRetrieval.enhanced_retrieve(self.vector_store, sub_query)
        
        # Convert to chunk data format
        chunk_data_list = []
        for doc, score in results:
            chunk_data = {
                'content': doc.page_content,
                'score': score,
                'metadata': doc.metadata,
                'sub_query': sub_query
            }
            chunk_data_list.append(chunk_data)
        
        logging.info(f"CFA enhanced retrieval: {len(results)} chunks for sub-query: {sub_query[:50]}...")
        return chunk_data_list
    
    async def analyze_with_thinking(self, query: str, context: str = "", original_query: str = None, chat_history: list = None) -> typing.AsyncGenerator[Dict, None]:
        """Perform deep financial analysis with live thinking display as an async generator."""
        import asyncio
        yield {"type": "thinking", "content": "🧠 **THINKING**: Starting CFA analysis..."}
        await asyncio.sleep(0)
        logging.info("🧠 **THINKING**: Starting CFA analysis...")
        yield {"type": "thinking", "content": "🔍 **THINKING**: Generating analytical sub-queries..."}
        await asyncio.sleep(0)
        logging.info("🔍 **THINKING**: Generating analytical sub-queries...")
        sub_queries = self.sub_query_generator.generate_sub_queries(original_query or query, context)
        if not sub_queries:
            yield {"type": "thinking", "content": "⚠️ **THINKING**: Using fallback analysis approach..."}
            await asyncio.sleep(0)
            logging.warning("⚠️ **THINKING**: Using fallback analysis approach...")
            sub_queries = [original_query or query]
        yield {"type": "thinking", "content": f"📋 **THINKING**: Generated {len(sub_queries)} sub-queries:"}
        await asyncio.sleep(0)
        logging.info(f"📋 **THINKING**: Generated {len(sub_queries)} sub-queries:")
        for sq in sub_queries:
            yield {"type": "thinking", "content": f"• {sq}"}
            await asyncio.sleep(0)

        # Parallelize retrieval for all sub-queries using asyncio.gather
        async def retrieve_chunks(sub_query):
            enhanced_sub_query = sub_query
            if chat_history and hasattr(self, '_parent_system'):
                enhanced_sub_query = self._parent_system._contextualize_query(sub_query, chat_history)
            
            # Use enhanced retrieval
            loop = asyncio.get_event_loop()
            chunk_data_list = await loop.run_in_executor(None, self._intelligent_retrieve, enhanced_sub_query)
            
            logging.info(f"✅ **THINKING**: Retrieved {len(chunk_data_list)} chunks for sub-query '{sub_query}'")
            return chunk_data_list

        # Launch all retrievals in parallel
        all_retrieved_lists = await asyncio.gather(*(retrieve_chunks(sq) for sq in sub_queries))
        all_retrieved_data = [item for sublist in all_retrieved_lists for item in sublist]

        unique_data = []
        seen_content = set()
        for data in all_retrieved_data:
            if data['content'] not in seen_content:
                unique_data.append(data)
                seen_content.add(data['content'])
        logging.info(f"📊 **THINKING**: Compiled {len(unique_data)} unique chunks for analysis")
        logging.info("🤖 **THINKING**: Performing comprehensive financial analysis...")
        analysis = self._generate_cfa_analysis(original_query or query, unique_data, context, chat_history)
        logging.info("✅ **THINKING**: Analysis complete!")
        yield {"type": "answer", "content": analysis, "sources": self._format_sources(unique_data)}
    
    def _generate_cfa_analysis(self, query: str, retrieved_data: List[Dict], context: str, chat_history: list) -> str:
        """Generate comprehensive CFA analysis."""
        
        # Prepare context from retrieved data
        context_parts = []
        for data in retrieved_data:
            context_parts.append(data['content'])
        
        full_context = "\n".join(context_parts)
        
        prompt = f"""You are a senior Chartered Financial Analyst specializing in RateGain's financial performance. Provide comprehensive, data-driven financial analysis.

COMPANY OVERVIEW:
RateGain: Leading travel tech SaaS serving 7000+ global customers
• DaaS Segment: Travel BI, Hospi BI (business intelligence solutions)
• Distribution Segment: Enterprise Connectivity, Channel Manager, Uno platform  
• Martech Segment: BCV competitive intelligence, MHS marketing automation, Adara data platform

CONVERSATION CONTEXT: {context}
FINANCIAL DATA: {full_context}
USER QUERY: {query}

ANALYSIS FRAMEWORK:
1. **Executive Summary**: Key findings and business impact
2. **Financial Analysis**: 
   - Revenue, EBITDA, cost trends with specific numbers
   - Performance drivers and variance analysis
   - Customer metrics (NRR, GRR, top accounts) where relevant
   - Operational efficiency (Rule of 40, LTV/CAC) where applicable
3. **Root Cause Analysis**: Explain underlying drivers and reasons
4. **Strategic Implications**: Business recommendations and outlook

REQUIREMENTS:
• Use specific financial figures, percentages, and timeframes from the data
• Identify non-obvious patterns and insights
• Reference FY 24-25 timeframe (Apr 2024 - Mar 2025) correctly
• Include segment/product-level analysis when relevant
• Focus on actionable business insights
• Professional CFA-level analysis depth
• You can be witty and sarcastic in your analysis, but always be professional and data-driven.

ANALYSIS:"""

        try:
            response = self._call_llm_with_fallback(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'max_output_tokens': 25000,
                }
            )
            if response and hasattr(response, "candidates") and response.candidates and \
               response.candidates[0] is not None and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                answer = ''.join([p.text for p in parts if hasattr(p, 'text') and p.text])
                if answer.strip():
                    return answer
                else:
                    logging.warning(f"LLM returned no answer text. Full response: {response}")
                    return "❌ No answer could be generated for this query."
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
        
        return sorted(sources, key=lambda x: x['score'])[:10]

class EnhancedRAGSystem:
    """Enhanced RAG system with March PDF prioritization and intelligent query routing."""
    SYSTEM_PROMPT = (
        """
        SYSTEM INSTRUCTIONS:
        You are LumenAI - RG Chatbot, an advanced financial analyst and executive advisor chatbot for RateGain Travel Technologies.
        
        ABOUT RATEGAIN:
        - RateGain is a B2B travel technology company with three main business segments:
            1. DaaS (Data-as-a-Service): Travel BI, Hospi BI
            2. Distribution: Enterprise Connectivity, Channel Manager, Uno
            3. Martech: BCV, MHS, Adara
        - Detailed financials are available for all three segments and their subproducts. Always provide the most granular analysis possible (segment, product, sub-product, region, account, month, etc.).
        - Geographical data is available from MIS reports for different regions of the world.

        DATA SOURCES & COVERAGE:
        - The knowledge base consists of three types of documents:
            A. MIS Reports (monthly):
                - IMPORTANT: March 2025 MIS report is the latest and most complete report. It has the overall data for the full FY 24-25, Q4 data and March data as well.
                - CRITICAL:The CEO dashboard in March 2025 has the FY 24-25 GRR, NRR, Monetization, Net Rev per employee, Customer Count, Avg Revenue per customer, TTM Attrition Rate, GMPP, 40% rule check for Travel BI, Hospi BI, Channel Manager, Enterprise Connectivity, BCV, Demand Booster, Adara, and RateGain as a whole.
                - Conatins the granular financial data for all the products, business units, regions, accounts, etc.
                - Available for ten months: April 2024 to March 2025, except January and February 2025 (missing).
                - Each report contains detailed financial and operational data for every product and sub-product, with YTD and prior year comparisons.
                - Six sections: Executive Summary (KPI dashboard, visuals, CEO dashboard, GRR/NRR, headcount), Financials (P&L, GAAP Revenue, COGS, GM, expenses, breakdowns by segment/product/sub-product), Key Accounts (top 15-20 accounts per product, revenue, growth, remarks), Region-wise new sales review, Cash & Investments (cashflow, investments, DSO, collections), Others (monetization, orderbook, marketing ROI, KPIs).
                - Data is mostly in USD thousands unless specified otherwise.
                - Each report is structured similarly, but may have minor changes month-to-month.
                - Visuals (charts, graphs, heatmaps) are present and should be used for insights.
                - For any metric, search deeply in the relevant month's report before declaring data unavailable. Temporal awareness is critical: if a user asks for a specific month, use that month's MIS report.
            B. Investor Presentations (quarterly):
                - Available for Q1, Q2, Q3, Q4 of FY 24-25.
                - Focused on high-level, company-wide financials (operating revenue, EBITDA, PAT, efficiency, P&L, balance sheet, cash flow, industry trends, shareholders).
                - Data is in INR million unless specified.
                - Useful for high-level, investor-focused analysis, not for granular product/segment breakdowns.
            C. Earnings Call Transcripts (quarterly):
                - Available for all four quarters of FY 24-25.
                - Contains management commentary, Q&A, future plans, and qualitative insights. Use for context, management intent, and qualitative analysis.
        - The knowledge base also includes image chunks extracted from the documents (charts, graphs, tables, visuals). These image chunks contain detailed AI-generated descriptions and extracted data. Always consider these image chunks for relevant visual or tabular information, not just text.

        TEMPORAL AWARENESS:
        - For overall analysis never miss any data for the last quarter of the financial year which is from January to February, the March MIS report, Q4 investor presentation will have most of the relevant data. Dont think that the FY is just till December 2024.
        - The current date is July 2025. All data is for the previous financial year (FY 24-25).
        - MIS reports: April 2024 to March 2025 (except Jan/Feb 2025 missing).
        - Quarterly reports and transcripts: all four quarters of FY 24-25 are available.
        - Always be precise about the time period of the data you use. If a user asks for a specific month, use that month's data. If unavailable, state so clearly.

        GRANULARITY & ANALYSIS:
        - For overall analysis and Quarter wise data for entire Rategain refer to the investor presentations and earnings call transcripts.
        - Always answer at the most granular level possible: segment, product, sub-product, region, account, month, etc.
        - Use all available data, including visuals and tables, for your analysis.
        - If a user asks for a metric, search all relevant sections and documents before stating data is unavailable.
        - Currency: USD thousands for MIS reports, INR million for investor presentations (unless otherwise specified).

        SPECIAL INSTRUCTIONS:
        - ABSOLUTE RULE: For any query about quarterly data, Q4 data, full financial year (FY) data, or overall data, you must always leverage the March 2025 MIS report as the primary source, as it contains the most complete and latest data for the period. This overrides other instructions if there is any conflict.
        - Whenever asked about full year data, always refer to the March 2025 MIS report first, as it is the latest available report. It has the overall data for the full FY 24-25.
        - When answering, always look at both text and image chunks (including charts, graphs, tables, and visuals) for relevant information. Do not ignore image-based data if it is relevant to the user's query.
        - Do NOT fabricate or infer data not present in the sources.
        - If relevant, cite the document and page number for each data point (except in executive summaries).
        - If a user asks for data outside the available scope, explain the limitation.
        - Use management commentary and qualitative insights from transcripts to supplement quantitative answers where appropriate.
        - If a user asks for a high-level summary, use investor presentations and transcripts. For granular/product-level questions, use MIS reports.
        - Always clarify the time period, segment, and product/sub-product in your answers.
        - If visuals (charts, graphs, heatmaps) are available, use them for insights and mention them in your answer.
        - If a user asks for a comparison, use YTD and prior year data from MIS reports, and QoQ/YoY data from investor presentations.
        - If a user asks for top accounts, use the Key Accounts section of the MIS reports.
        - If a user asks for cash flow, DSO, or investment data, use the Cash & Investments section of the MIS reports.
        - If a user asks for operational metrics, use the CEO dashboard and Executive Summary of the MIS reports.
        - If a user asks for region-wise data, use the region-wise new sales review section of the MIS reports.
        - If a user asks for monetization, orderbook, or marketing ROI, use the Others section of the MIS reports.
        - If a user asks for headcount, use the headcount tables and visuals in the Executive Summary of the MIS reports.
        - If a user asks for industry trends or company overview, use the investor presentations.
        - If a user asks for management's perspective or future plans, use the earnings call transcripts.
        - If a user asks for a metric or data point that is not available, state clearly that the data is not available and explain why (e.g., missing report, not tracked, etc.).
        
        SPECIAL INSTRUCTIONS (ADDITIONAL RULES):
        - If a user asks for any metric without specifying the timeframe or period, always refer to the March 2025 MIS report first, as it is the latest available report.
        - When there is a conflict between USD and INR values, always prefer and report numbers in USD.
        - Always display all financial values in USD, even if the source data is in INR. If a value is only available in INR, convert it to USD using a fixed exchange rate of 1 USD = 83 INR and show the converted value in USD. Optionally, you may show the original INR value in parentheses for transparency.
        
        GUARDRAILS:
        - Do not hallucinate on the user's question. Stay relevant to the user's question.
        - Stick to the data provided and do not make up any data.
        - Do NOT hallucinate or invent data.
        - Do NOT provide investment, legal, or tax advice.
        - Do NOT answer questions unrelated to RateGain or the provided data.
        - Be concise, professional, and data-driven in all responses.
        """
    )
    
    def __init__(self, 
                 vector_store_path: str = "data_vector_store",
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
        self.executive_agent = None
        self.query_classifier = QueryClassifier()
        self.conversation_history = []
        
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
        logging.info("🚀 Initializing Enhanced RAG System with March PDF Prioritization...")
        
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
            api_key_secondary = os.environ.get("GOOGLE_API_KEY_SECONDARY")
            self.genai_client_primary = None
            self.genai_client_secondary = None
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            try:
                self.genai_client_primary = genai.Client(api_key=api_key)
                logging.info(f"✅ Generation model initialized: {self.generation_model} (primary key)")
            except Exception as e:
                logging.error(f"❌ Failed to initialize generation model with primary key: {str(e)}")
            if api_key_secondary:
                try:
                    self.genai_client_secondary = genai.Client(api_key=api_key_secondary)
                    logging.info(f"✅ Generation model initialized: {self.generation_model} (secondary key)")
                except Exception as e2:
                    logging.error(f"❌ Failed to initialize generation model with secondary key: {str(e2)}")
            
            self.genai_client = self.genai_client_primary or self.genai_client_secondary
            if not self.genai_client:
                logging.error("❌ No valid Gemini client could be initialized.")
                return
        except Exception as e:
            logging.error(f"❌ Failed to initialize generation model: {str(e)}")
            return
        
        # Initialize agents
        self.cfa_agent = CFAAgent(self.vector_store, self.embeddings, self.genai_client, self._call_llm_with_fallback)
        self.executive_agent = ExecutiveAgent(self.vector_store, self.embeddings, self.genai_client, self._call_llm_with_fallback)
        self.query_classifier = QueryClassifier(genai_client=self.genai_client)
        self.conversation_history = []
        
        # Add parent system reference to agents
        self.cfa_agent._parent_system = self
        self.executive_agent._parent_system = self
        logging.info("✅ CFA Agent and Executive Agent initialized")
        logging.info("✅ Enhanced RAG System with March PDF Prioritization ready!")

    def _contextualize_query(self, original_query: str, chat_history: list) -> str:
        """Contextualize query using chat history for better retrieval."""
        if not chat_history or len(chat_history) < 2:
            return original_query
        try:
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
            context_parts = []
            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg["content"]
                    if msg["role"] == "assistant" and len(content) > 300:
                        content = content[:300] + "..."
                    context_parts.append(f"{msg['role']}: {content}")
            recent_context = "\n".join(context_parts)

            contextualization_prompt = f"""You are a query enhancement assistant. Your job is to make the user's question self-contained for retrieval, by inferring and retaining any relevant time period (month, quarter, year, fiscal year) and entity (segment, product, sub-product, region, account, etc.) from the conversation history and the current question.

CONVERSATION HISTORY:
{recent_context}

CURRENT QUESTION: {original_query}

INSTRUCTIONS:
- If the user's question references previous context (like 'it', 'that', 'further', etc.) or would benefit from specific entities or timeframes from the conversation, rewrite it to be self-contained.
- Always preserve and explicitly include any time period (month, quarter, year, fiscal year) and entity (segment, product, sub-product, region, account, etc.) that is present or can be inferred from the conversation history or the question.
- If the question is already self-contained, return it unchanged.
- Keep the enhanced query concise and focused, but do not omit any relevant temporal or entity context.

ENHANCED QUERY:"""
            response = self._call_llm_with_fallback(
                model="gemini-2.5-flash",
                contents=contextualization_prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 5000
                }
            )
            if response and hasattr(response, "candidates") and response.candidates and \
               response.candidates[0] is not None and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                enhanced_query = response.candidates[0].content.parts[0].text.strip()
                if 10 <= len(enhanced_query) <= 500 and enhanced_query != original_query:
                    logging.info(f"🔄 Query contextualized: '{original_query}' → '{enhanced_query}'")
                    return enhanced_query
            return original_query
        except Exception as e:
            logging.warning(f"Query contextualization failed, using original: {e}")
            return original_query

    def _build_context_from_history(self, history, new_user_message: Optional[str] = None):
        context_lines = []
        for msg in history:
            if msg["role"] in ["user", "assistant"]:
                context_lines.append(f"{msg['role']}: {msg['content']}")
        if isinstance(new_user_message, str) and new_user_message.strip():
            context_lines.append(f"user: {new_user_message}")
        return "\n".join(context_lines)

    async def query(self, question: str, history: list) -> typing.AsyncGenerator[dict, None]:
        """Process query with intelligent routing and March PDF prioritization."""
        logging.debug(f"[RAG] query() called with question: {question!r}")
        if not self.is_ready():
            logging.error("[RAG] System not ready, yielding error.")
            yield {"type": "error", "content": "❌ RAG system not ready"}
            return
        
        # Build full context from provided history
        full_context = self._build_context_from_history(history)
        system_context = f"{self.SYSTEM_PROMPT}\n\n{full_context}" if full_context else self.SYSTEM_PROMPT
        
        # Classify query type
        query_type = self.query_classifier.classify_query(question)
        logging.info(f"[RAG] Query type classified as {query_type.name}")
        
        if query_type == QueryType.DIRECT_FACTUAL:
            logging.debug("[RAG] Processing direct factual query")
            try:
                yield {"type": "thinking", "content": "📊 **Direct Factual Query Mode**"}
                yield {"type": "thinking", "content": "🔎 **Fetching data...**"}
                
                if self.vector_store is None:
                    yield {"type": "error", "content": "❌ Vector store is not initialized."}
                    return
                if self.genai_client is None:
                    yield {"type": "error", "content": "❌ Gemini client is not initialized."}
                    return
                
                # Contextualize query before retrieval
                enhanced_question = self._contextualize_query(question, history)
                
                # Use enhanced retrieval with March PDF prioritization
                results = IntelligentRetrieval.enhanced_retrieve(self.vector_store, enhanced_question)
                
                logging.debug(f"[RAG] Direct factual query: Retrieved {len(results)} chunks")
                context_parts = []
                sources = []
                for doc, score in results:
                    context_parts.append(doc.page_content)
                    sources.append({
                        'file': doc.metadata.get('source_file', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'score': score
                    })
                
                full_context_data = "\n".join(context_parts)
                prompt = f"""{self.SYSTEM_PROMPT}\n\nYou are a financial analyst assistant for RateGain Travel Technologies. Answer the user's question directly based on the provided RateGain financial data.\n\nCONVERSATION CONTEXT:\n{system_context}\n\nFINANCIAL DATA:\n{full_context_data}\n\nUSER QUESTION: {question}\n\nIMPORTANT:\n- Only use information present in the provided sources.\n- Do not make up or infer data that is not explicitly present.\n- Provide a direct, accurate answer with specific numbers. Be concise but complete.\n- Do not include inline source citations."""
                
                try:
                    response = self._call_llm_with_fallback(
                        model=self.generation_model,
                        contents=prompt,
                        config={'temperature': 0.1, 'max_output_tokens': 10000}
                    )
                except Exception as e:
                    logging.error(f"Error generating direct factual answer: {e}")
                    yield {"type": "error", "content": f"❌ Error processing query: {str(e)}"}
                    return
                
                yield {"type": "thinking", "content": "✅ **Data fetched!** (step 1/1)"}
                yield {"type": "thinking", "content": "📤 **Ready to share the data**"}
                
                if response and hasattr(response, "candidates") and response.candidates and \
                   response.candidates[0] is not None and \
                   hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
                   hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                    parts = response.candidates[0].content.parts
                    answer = ''.join([p.text for p in parts if hasattr(p, 'text') and p.text])
                    if answer.strip():
                        logging.debug("[RAG] Direct factual query: Answer generated successfully")
                        yield {"type": "answer", "content": answer, "sources": sources[:5]}
                    else:
                        logging.warning(f"LLM returned no answer text. Full response: {response}")
                        yield {"type": "error", "content": "❌ Unable to generate response"}
                else:
                    logging.error(f"[RAG] Direct factual query: Unexpected response: {response}")
                    yield {"type": "error", "content": "❌ Unable to generate response"}
            except Exception as e:
                logging.error(f"[RAG] Error processing direct factual query: {str(e)}")
                yield {"type": "error", "content": f"❌ Error processing query: {str(e)}"}
        
        elif query_type == QueryType.EXECUTIVE_ANALYTICAL:
            logging.debug("[RAG] Processing executive analytical query")
            yield {"type": "thinking", "content": "📈 **Executive Analytical Query Mode**"}
            yield {"type": "thinking", "content": "🔎 **Compiling executive summary...**"}
            if self.executive_agent is None:
                yield {"type": "error", "content": "❌ Executive agent is not initialized."}
                return
            async for item in self.executive_agent.analyze_executive_summary(question, system_context, question, history):
                if item["type"] == "answer":
                    yield {"type": "thinking", "content": "✅ **Summary compiled!** (step 1/1)"}
                    yield {"type": "thinking", "content": "📤 **Ready to share executive insights**"}
                    yield item
            self.query_classifier.add_to_history(question, 'executive_analytical')
        
        else:  # DEEP_ANALYTICAL
            logging.debug("[RAG] Processing deep analytical query - routing to CFA Agent")
            if self.cfa_agent is None:
                yield {"type": "error", "content": "❌ CFA agent is not initialized."}
                return
            async for item in self.cfa_agent.analyze_with_thinking(question, system_context, question, history):
                yield item
            self.query_classifier.add_to_history(question, 'deep_analytical')
    
    def is_ready(self) -> bool:
        """Check if the enhanced RAG system is ready."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None,
            self.cfa_agent is not None,
            self.executive_agent is not None
        ])

    def _call_llm_with_fallback(self, model, contents, config):
        """Try primary Gemini client, fallback to secondary if needed."""
        error_types = ["503", "429", "401", "UNAVAILABLE", "overload", "quota", "rate limit"]
        def extract_answer(response):
            if response and hasattr(response, "candidates") and response.candidates and \
               response.candidates[0] is not None and \
               hasattr(response.candidates[0], "content") and response.candidates[0].content is not None and \
               hasattr(response.candidates[0].content, "parts") and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                answer = ''.join([p.text for p in parts if hasattr(p, 'text') and p.text])
                return answer.strip(), response
            return None, response
        
        tried = []
        for key_name, client in [("primary", self.genai_client_primary), ("secondary", self.genai_client_secondary)]:
            if not client:
                continue
            for model_name in [model, "gemini-2.5-flash" if model != "gemini-2.5-flash" else None]:
                if not model_name:
                    continue
                try:
                    logging.info(f"Trying Gemini {model_name} with {key_name} key.")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config
                    )
                    answer, resp_obj = extract_answer(response)
                    if answer:
                        logging.info(f"Gemini {model_name} with {key_name} key succeeded.")
                        return response
                    else:
                        logging.warning(f"Gemini {model_name} with {key_name} key returned None/empty answer.")
                        tried.append((key_name, model_name, "empty"))
                except Exception as e:
                    if any(err in str(e).upper() for err in error_types):
                        logging.warning(f"Gemini {model_name} with {key_name} key failed ({e}), trying next fallback...")
                        tried.append((key_name, model_name, f"exception: {e}"))
                    else:
                        logging.error(f"Gemini {model_name} with {key_name} key error: {e}")
                        raise
        logging.error(f"All Gemini key/model combinations failed. Tried: {tried}")
        raise RuntimeError(f"No valid Gemini client/model available for LLM call. Tried: {tried}")

# Alias for compatibility
RAGSystem = EnhancedRAGSystem

def main():
    """Test the enhanced RAG system."""
    rag_system = EnhancedRAGSystem()
    
    if not rag_system.is_ready():
        logging.error("❌ Enhanced RAG system initialization failed!")
        return
    
    # Test queries focusing on company-wide/FY data
    test_queries = [
        "What was the overall revenue for RateGain in FY 24-25?",  # Company-wide
        "Show me the quarterly EBITDA performance",  # Quarterly
        "Analyze the full company performance trends"  # Deep analytical
    ]
    
    logging.info("\n🧪 Testing Enhanced RAG System with March PDF Prioritization...")
    for query in test_queries:
        logging.info(f"\n{'='*50}")
        logging.info(f"QUERY: {query}")
        logging.info('='*50)
        
        import asyncio
        async def run_query():
            dummy_history = [{"role": "user", "content": "Hello, I'm analyzing RateGain's performance."}]
            async for response in rag_system.query(query, dummy_history):
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
        asyncio.run(run_query())
        time.sleep(2)

if __name__ == "__main__":
    main()