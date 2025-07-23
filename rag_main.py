# Enhanced RAG System with CFA Agent and Executive Agent

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
        self.conversation_history = deque(maxlen=5)  # Track last 5 turns
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
        # 1. Deep Analytical: root cause, why, drivers, analysis, etc.
        deep_analysis_keywords = [
            'root cause', 'why', 'explain', 'driver', 'reason', 'cause', 'deep analysis', 'detailed analysis',
            'elaborate', 'breakdown', 'comprehensive analysis', 'in-depth', 'thorough', 'analyze', 'analyse',
            'further analysis', 'drill down', 'expand on', 'dive deeper', 'what caused', 'explain the reason', 'drivers'
        ]
        if any(kw in query_lower for kw in deep_analysis_keywords):
            return QueryType.DEEP_ANALYTICAL
        # 2. Executive Analytical: summary, overview, compare, trend, performance, insight, high level, etc.
        executive_keywords = [
            'summary', 'summarize', 'overview', 'insight', 'compare', 'comparison', 'trend', 'performance',
            'high level', 'business impact', 'implication', 'implications', 'key findings', 'key insights',
            'business insight', 'business overview', 'business summary', 'business review', 'review', 'synthesis',
            'recap', 'conclusion', 'conclude', 'highlight', 'main points', 'main findings', 'main insights', 'synthesize'
        ]
        if any(kw in query_lower for kw in executive_keywords):
            return QueryType.EXECUTIVE_ANALYTICAL
        # 3. Direct Factual: must contain metric/number keywords and NOT contain summary/analysis words
        metric_keywords = [
            'revenue', 'ebitda', 'cost', 'costs', 'profit', 'loss', 'nrr', 'grr', 'retention', 'monetization',
            'department spending', 'sales', 'ltv', 'cogs', 'cashflow', 'cash flow', 'collection', 'accounts',
            'account', 'customer', 'customers', 'number of', 'amount', 'total', 'value', 'figure', 'metric', 'data',
            'percentage', 'percent', 'ratio', 'score', 'count', 'average', 'mean', 'median', 'variance', 'change',
            'increase', 'decrease', 'drop', 'rise', 'growth', 'decline', 'month', 'quarter', 'year', 'period', 'date',
            'as of', 'for', 'in', 'during', 'between', 'show', 'list', 'give', 'provide', 'display', 'report', 'state',
            'what is', 'what was', 'how many', 'how much', 'when', 'which', 'find', 'identify', 'fetch', 'extract'
        ]
        # Exclude if summary/analysis words present
        if any(kw in query_lower for kw in metric_keywords) and not any(kw in query_lower for kw in executive_keywords + deep_analysis_keywords):
            return QueryType.DIRECT_FACTUAL
        # Default: executive analytical (catch-all for high-level)
        return QueryType.EXECUTIVE_ANALYTICAL

class ExecutiveAgent:
    """Executive-level analytical agent for concise, comprehensive insights."""
    
    def __init__(self, vector_store, embeddings, genai_client, llm_fallback_fn):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.genai_client = genai_client
        self._call_llm_with_fallback = llm_fallback_fn
    
    async def analyze_executive_summary(self, query: str, context: str = "", original_query: str = None, chat_history: list = None) -> typing.AsyncGenerator[Dict, None]:
        """Perform executive-level analysis with concise insights."""
        import asyncio
        
        # Do not yield thinking steps for executive agent, only yield the final answer
        try:
            # Contextualize query before retrieval if chat history provided
            enhanced_query = query
            if chat_history and hasattr(self, '_parent_system'):
                enhanced_query = self._parent_system._contextualize_query(original_query or query, chat_history)
            # Retrieve relevant data using enhanced query
            results = self.vector_store.similarity_search_with_score(enhanced_query, k=20)
            
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
        
        prompt = f"""You are a senior executive advisor for RateGain Travel Technologies, a global provider of SaaS solutions for travel and hospitality industry. Provide a concise executive summary (200-300 words) with key insights.

ABOUT RATEGAIN:
RateGain is a leading travel technology company serving 7000+ customers globally across hotels, airlines, car rentals, cruise lines, and travel agencies. The company operates through three main business segments:

1. **DaaS (Data-as-a-Service)**: 
   - Travel BI: Business intelligence for travel companies
   - Hospi BI: Business intelligence for hospitality sector

2. **Distribution**: 
   - Enterprise Connectivity: Channel management solutions
   - Channel Manager: Distribution channel optimization
   - Uno: Unified booking platform

3. **Martech (Marketing Technology)**:
   - BCV (Brand Compete View): Competitive intelligence
   - MHS (Marketing Hub Solutions): Marketing automation
   - Adara: Data-driven marketing platform

CONVERSATION CONTEXT:
{conversation_context}

FINANCIAL DATA:
{retrieved_context}

USER QUERY: {query}

IMPORTANT:
- Only use information present in the provided sources.
- Do not make up or infer data that is not explicitly present.
- Your summary must be concise but highly insightful.
- Surface hidden, non-obvious, or counterintuitive insights that a typical reader might overlook.
- Highlight patterns, anomalies, or trends that are not immediately apparent.

EXECUTIVE SUMMARY REQUIREMENTS:
- Length: 200-300 words maximum
- Format: Use bullet points for key insights
- Use tables for comparative analysis when comparing multiple items
- Include specific metrics and percentages
- Focus on business impact and actionable insights
- Highlight at least one insight that is not obvious or is counterintuitive
- No inline source citations
- Structure: Brief overview, key insights (3-5 bullet points), business implications

RESPONSE FORMAT:
## Executive Summary
[2-3 sentence overview]

## Key Insights
• [Key finding 1 with specific metrics]
• [Key finding 2 with trend analysis]
• [Key finding 3 with business impact]
• [Non-obvious or hidden insight]

## Business Implications
[Brief strategic implications]

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
        prompt = f"""You are a Chartered Financial Analyst. Given the user's analytical query about RateGain financial data, generate a list of specific sub-queries that need to be answered to provide a comprehensive analysis.

ABOUT RATEGAIN:
RateGain is a leading travel technology company serving 7000+ customers globally across hotels, airlines, car rentals, cruise lines, and travel agencies. The company operates through three main business segments:

1. **DaaS (Data-as-a-Service)**: 
   - Travel BI: Business intelligence for travel companies
   - Hospi BI: Business intelligence for hospitality sector

2. **Distribution**: 
   - Enterprise Connectivity: Channel management solutions
   - Channel Manager: Distribution channel optimization
   - Uno: Unified booking platform

3. **Martech (Marketing Technology)**:
   - BCV (Brand Compete View): Competitive intelligence
   - MHS (Marketing Hub Solutions): Marketing automation
   - Adara: Data-driven marketing platform

AVAILABLE DATA: Revenue, EBITDA, Costs, Top Accounts, NRR (Net Revenue Retention), GRR (Gross Revenue Retention), Retention, Monetization, Department Spending, "rule of 40", sales multiple, LTV2CAC (LTV to CAC ratio)
TIME PERIOD: April 2024 - March 2025

CONVERSATION CONTEXT:
{context}

USER QUERY: {original_query}

Generate 8-10 specific sub-queries that will help analyze this comprehensively. Focus especially on the following metrics as per relevance:
- Base metrics (EBITDA, Revenue for specific periods)
- Comparative analysis if multiple periods/products mentioned
- NRR (Net Revenue Retention) and GRR (Gross Revenue Retention)
- Top accounts (found in the top accounts section for each product), Department Spending, COGS, Monetization
- "Rule of 40", Sales multiple, LTV2CAC (LTV to CAC ration)
- Investment Summary, Cashflow, M-o-M Cash Movement, Collection, Day of sales outstanding
- Monetisation for different products and services

Only generate sub-queries that are directly relevant to the user's query and the provided business context. Do NOT go off topic or include unrelated financial concepts.

Return only the sub-queries, one per line, without numbering or explanations."""
        
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
                return sub_queries[:8]  # Limit to 8 sub-queries
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

        # --- Parallelize retrieval for all sub-queries using asyncio.gather ---
        async def retrieve_chunks(sub_query):
            enhanced_sub_query = sub_query
            if chat_history and hasattr(self, '_parent_system'):
                enhanced_sub_query = self._parent_system._contextualize_query(sub_query, chat_history)
            
            # If your vector store has an async API, use await here. Otherwise, run in executor.
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.vector_store.similarity_search_with_score, enhanced_sub_query, 15)
            
            chunk_data_list = []
            for doc, score in results:
                chunk_data = {
                    'content': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata,
                    'sub_query': sub_query
                }
                chunk_data_list.append(chunk_data)  # ✅ MOVED INSIDE the for loop
            
            logging.info(f"✅ **THINKING**: Retrieved {len(chunk_data_list)} chunks for sub-query '{sub_query}'")
            return chunk_data_list

        # Launch all retrievals in parallel
        all_retrieved_lists = await asyncio.gather(*(retrieve_chunks(sq) for sq in sub_queries))
        all_retrieved_data = [item for sublist in all_retrieved_lists for item in sublist]
        # --- End parallelization ---

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
        
        prompt = f"""You are a senior Chartered Financial Analyst (CFA) specializing in RateGain's financial performance. Provide a comprehensive financial analysis based on the data provided.

ABOUT RATEGAIN:
RateGain is a leading travel technology company serving 7000+ customers globally across hotels, airlines, car rentals, cruise lines, and travel agencies. The company operates through three main business segments:

1. **DaaS (Data-as-a-Service)**: 
   - Travel BI: Business intelligence for travel companies
   - Hospi BI: Business intelligence for hospitality sector

2. **Distribution**: 
   - Enterprise Connectivity: Channel management solutions
   - Channel Manager: Distribution channel optimization
   - Uno: Unified booking platform

3. **Martech (Marketing Technology)**:
   - BCV (Brand Compete View): Competitive intelligence
   - MHS (Marketing Hub Solutions): Marketing automation
   - Adara: Data-driven marketing platform

CONVERSATION CONTEXT:
{context}

FINANCIAL DATA:
{full_context}

USER QUERY: {query}

ANALYSIS REQUIREMENTS:
1. **Executive Summary**: Start with key findings
2. **Detailed Financial Analysis**: (Include the data points mentioned below as per relevance)
   - Analyze EBITDA, Revenue, Costs systematically
   - Identify trends, variances, and performance drivers
   - Examine top accounts and customer dynamics
   - Review department spending patterns
   - Emphasize NRR (Net Revenue Retention), GRR (Gross Revenue Retention), Retention, top accounts, "rule of 40", sales multiple, and LTV2CAC (LTV to CAC ratio) wherever relevant
   - Investment Summary, Cashflow, M-o-M Cash Movement, Collection, Day of sales outstanding whatever is relevant
   - Monetisation for different products and services
3. **Root Cause Analysis**: Explain the "why" behind numbers
4. **Business Implications**: What this means for RateGain
5. **Data-Driven Insights**: Include specific numbers, percentages, and comparisons

IMPORTANT:
- Be factually accurate with all numbers
- Reference specific time periods correctly (FY 2024-25: Apr 2024 - Mar 2025)
- Provide actionable business insights
- Use professional financial analysis language
- Include specific account names and financial figures when available
- Do not include inline source citations
- If relevant, discuss NRR, GRR, Retention, top accounts, "rule of 40", sales multiple, and LTV2CAC in your analysis

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
               hasattr(response.candidates[0], "content") and response.candidates[0].content.parts:
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
        
        return sorted(sources, key=lambda x: x['score'])[:10]  # Top 10 sources

class EnhancedRAGSystem:
    """Enhanced RAG system with executive agent, CFA agent and intelligent query routing."""
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
        self.conversation_history = []  # Store full chat history as list of dicts
        
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
        logging.info("🚀 Initializing Enhanced RAG System with Executive Agent...")
        
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
            # Use primary as default for compatibility
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
        self.conversation_history = []  # Store full chat history as list of dicts
        # Add parent system reference to agents
        self.cfa_agent._parent_system = self
        self.executive_agent._parent_system = self
        logging.info("✅ CFA Agent and Executive Agent initialized")
        logging.info("✅ Enhanced RAG System ready!")

    def _contextualize_query(self, original_query: str, chat_history: list) -> str:
        """
        Contextualize query using chat history for better retrieval.
        Always uses last 4 messages to decide and enhance if needed.
        Leverage the LLM to infer and retain time period and entity context, without hardcoded extraction.
        """
        # Skip if no meaningful history
        if not chat_history or len(chat_history) < 2:
            return original_query
        try:
            # Extract recent conversation (last 4 messages)
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
            # Build conversation context
            context_parts = []
            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"]:
                    content = msg["content"]
                    if msg["role"] == "assistant" and len(content) > 300:
                        content = content[:300] + "..."
                    context_parts.append(f"{msg['role']}: {content}")
            recent_context = "\n".join(context_parts)

            # LLM prompt: let the LLM infer and retain time period/entity context
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
                # Use enhanced query if it's different and reasonable
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
        """Process query with intelligent routing as an async generator. Accepts external chat history."""
        logging.debug(f"[RAG] query() called with question: {question!r}, history type: {type(history)}, history: {history}")
        if not self.is_ready():
            logging.error("[RAG] System not ready, yielding error.")
            yield {"type": "error", "content": "❌ RAG system not ready"}
            return
        # Build full context from provided history
        full_context = self._build_context_from_history(history)
        logging.debug(f"[RAG] Built full_context: {full_context!r}")
        # Prepend system prompt to context for all LLM calls
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
                results = self.vector_store.similarity_search_with_score(enhanced_question, k=20)
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
                    logging.error(f"Error generating direct factual answer (both keys failed): {e}")
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
                        logging.debug("[RAG] Direct factual query: Answer generated successfully, yielding answer.")
                        yield {"type": "answer", "content": answer, "sources": sources[:5]}
                    else:
                        logging.warning(f"LLM returned no answer text. Full response: {response}")
                        yield {"type": "error", "content": "❌ Unable to generate response"}
                else:
                    logging.error(f"[RAG] Direct factual query: Unexpected response: {response}")
                    if response and hasattr(response, "candidates") and response.candidates and response.candidates[0] is not None and hasattr(response.candidates[0], "content"):
                        logging.debug(f"[RAG] candidates[0].content: {response.candidates[0].content}")
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
                logging.debug(f"[RAG] Yielding executive agent item: {item}")
                if item["type"] == "answer":
                    yield {"type": "thinking", "content": "✅ **Summary compiled!** (step 1/1)"}
                    yield {"type": "thinking", "content": "📤 **Ready to share executive insights"}
                    yield item
            self.query_classifier.add_to_history(question, 'executive_analytical')
        
        else:  # DEEP_ANALYTICAL
            logging.debug("[RAG] Processing deep analytical query - routing to CFA Agent")
            if self.cfa_agent is None:
                yield {"type": "error", "content": "❌ CFA agent is not initialized."}
                return
            async for item in self.cfa_agent.analyze_with_thinking(question, system_context, question, history):
                logging.debug(f"[RAG] Yielding CFA agent item: {item}")
                yield item
            self.query_classifier.add_to_history(question, 'deep_analytical')
        logging.debug("[RAG] query() exiting.")
    
    def is_ready(self) -> bool:
        """Check if the enhanced RAG system is ready."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None,
            self.cfa_agent is not None,
            self.executive_agent is not None
        ])

    # Utility function for robust LLM call with per-request fallback
    def _call_llm_with_fallback(self, model, contents, config):
        """Try primary Gemini client, fallback to secondary if needed, including on None/empty answer. Log all key/model switches and empty answers."""
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
        # Try all (key, model) combinations: (primary, pro), (primary, flash), (secondary, pro), (secondary, flash)
        tried = []
        for key_name, client in [("primary", self.genai_client_primary), ("secondary", self.genai_client_secondary)]:
            if not client:
                continue
            for model_name in [model, "gemini-2.5-flash" if model != "gemini-2.5-flash" else None]:
                if not model_name:
                    continue
                try:
                    log_msg = f"Trying Gemini {model_name} with {key_name} key."
                    logging.info(log_msg)
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
                        logging.warning(f"Gemini {model_name} with {key_name} key returned None/empty answer. Full response: {resp_obj}")
                        tried.append((key_name, model_name, "empty"))
                except Exception as e:
                    if any(err in str(e).upper() for err in error_types):
                        logging.warning(f"Gemini {model_name} with {key_name} key failed ({e}), trying next fallback...")
                        tried.append((key_name, model_name, f"exception: {e}"))
                    else:
                        logging.error(f"Gemini {model_name} with {key_name} key error: {e}")
                        raise
        logging.error(f"All Gemini key/model combinations failed or returned empty. Tried: {tried}")
        raise RuntimeError(f"No valid Gemini client/model available for LLM call, or all returned empty/None answer. Tried: {tried}")

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
        "Compare the EBITDA for Hospi BI and Travel BI in Q2 and Q3",  # Executive
        "Analyze further the EBITDA trends for Hospi BI"  # Deep (follow-up)
    ]
    
    logging.info("\n🧪 Testing Enhanced RAG System...")
    for query in test_queries:
        logging.info(f"\n{'='*50}")
        logging.info(f"QUERY: {query}")
        logging.info('='*50)
        
        # Use regular for loop since main() is not async
        import asyncio
        async def run_query():
            # Create a dummy history for testing
            dummy_history = [{"role": "user", "content": "Hello, I'm a user."}]
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