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
from datetime import datetime
import dateutil.parser
import json
import openai

# Load environment variables
load_dotenv()

# Simulated data_processing module
class Config:
    vector_store_path = "data/vector_store"
    collection_name = "pdf_documents"

def extract_period_from_filename_simple(filename: str) -> dict:
    # Remove extension
    name = filename.rsplit('.', 1)[0]
    # Match patterns like may_2024, march_2025, etc.
    match = re.match(r"([a-zA-Z]+)[_ ](\d{4})", name)
    if match:
        month = match.group(1).capitalize()
        year = match.group(2)
        # Map month to quarter
        month_to_quarter = {
            "January": "Q4", "February": "Q4", "March": "Q4",
            "April": "Q1", "May": "Q1", "June": "Q1",
            "July": "Q2", "August": "Q2", "September": "Q2",
            "October": "Q3", "November": "Q3", "December": "Q3"
        }
        quarter = month_to_quarter.get(month, "Unknown")
        return {"month": month, "year": year, "quarter": quarter, "fiscal_year": "Unknown"}
    else:
        return {"month": "Unknown", "year": "Unknown", "quarter": "Unknown", "fiscal_year": "Unknown"}

def preprocess_document(doc_content: str, filename: str, metadata: Dict) -> List[Dict]:
    """Preprocess document with semantic chunking and temporal encoding using simple filename extraction."""
    chunks = []
    current_chunk = ""
    current_metadata = metadata.copy()

    # Use simple regex extraction for period info
    period_info = extract_period_from_filename_simple(filename)
    month = period_info.get("month", "Unknown")
    year = period_info.get("year", "Unknown")
    quarter = period_info.get("quarter", "Unknown")
    fiscal_year = period_info.get("fiscal_year", "Unknown")

    # Map month to quarter
    month_to_quarter = {
        "April": "Q1", "May": "Q1", "June": "Q1",
        "July": "Q2", "August": "Q2", "September": "Q2",
        "October": "Q3", "November": "Q3", "December": "Q3",
        "March": "Q4"  # Consolidated sheet
    }
    # quarter = month_to_quarter.get(month, "Unknown") # This line is now redundant as quarter is set above

    # Split content by sections and tables
    lines = doc_content.split("\n")
    chunk_id = 0
    table_mode = False
    table_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect table start/end
        if re.search(r'\|.*\|', line) or "Table of Content" in line or SmartChunkMerger.detect_table_content(line):
            table_mode = True
            table_content.append(line)
        else:
            if table_mode and table_content:
                # Save table chunk
                chunks.append({
                    "content": "\n".join(table_content),
                    "metadata": {
                        **current_metadata,
                        "chunk_id": f"{filename}_chunk_{chunk_id}",
                        "month": month,
                        "quarter": quarter,
                        "fiscal_year": fiscal_year,
                        "content_type": "table"
                    }
                })
                chunk_id += 1
                table_content = []
                table_mode = False
            
            # Handle non-table content
            if len(current_chunk) + len(line) < 1000:  # Arbitrary chunk size limit
                current_chunk += line + "\n"
            else:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        **current_metadata,
                        "chunk_id": f"{filename}_chunk_{chunk_id}",
                        "month": month,
                        "quarter": quarter,
                        "fiscal_year": fiscal_year,
                        "content_type": "text"
                    }
                })
                chunk_id += 1
                current_chunk = line + "\n"
    
    # Save remaining content
    if current_chunk:
        chunks.append({
            "content": current_chunk.strip(),
            "metadata": {
                **current_metadata,
                "chunk_id": f"{filename}_chunk_{chunk_id}",
                "month": month,
                "quarter": quarter,
                "fiscal_year": fiscal_year,
                "content_type": "text"
            }
        })
    if table_content:
        chunks.append({
            "content": "\n".join(table_content),
            "metadata": {
                **current_metadata,
                "chunk_id": f"{filename}_chunk_{chunk_id}",
                "month": month,
                "quarter": quarter,
                "fiscal_year": fiscal_year,
                "content_type": "table"
            }
        })
    
    return chunks

def run_data_processing(documents: List[Dict]) -> Tuple[List, List, Chroma]:
    """Simulated data processing pipeline with semantic chunking."""
    print("[RAG] Starting document processing...")
    all_chunks = []
    for doc in documents:
        filename = doc.get("filename", "unknown.pdf")
        print(f"[RAG] Processing document: {filename}")
        pages = doc.get("pages", [])
        for page_num, page_content in enumerate(pages, 1):
            print(f"[RAG]  - Chunking page {page_num} of {filename}")
            chunks = preprocess_document(
                page_content.get("content", ""),
                filename,
                {"source_file": filename, "page": page_num},
            )
            print(f"[RAG]    - {len(chunks)} chunks created for page {page_num}")
            all_chunks.extend(chunks)
    print(f"[RAG] Total chunks created: {len(all_chunks)}")
    # Save all chunks to a JSON file
    try:
        with open("all_chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print("[RAG] All chunks saved to all_chunks.json")
    except Exception as e:
        print(f"[RAG] Failed to save all_chunks.json: {e}")
    # Initialize Chroma vector store
    client = chromadb.PersistentClient(path=Config.vector_store_path)
    collection = client.get_or_create_collection(Config.collection_name)
    embeddings = GeminiEmbeddings("gemini-embedding-exp-03-07")
    print(f"[RAG] Starting embedding and vector store creation for {len(all_chunks)} chunks...")
    start_time = time.time()
    for idx, chunk in enumerate(all_chunks):
        if idx % 50 == 0:
            print(f"[RAG] Embedding chunk {idx+1}/{len(all_chunks)}")
        print(f"[RAG] [Embedding] Chunk {idx+1}/{len(all_chunks)}: {chunk['metadata'].get('chunk_id', idx)}")
        embedding = embeddings.embed_query(chunk["content"])
        if not embedding:
            print(f"[RAG] Skipping chunk {chunk['metadata'].get('chunk_id', idx)} due to empty embedding.")
            continue
        try:
            collection.add(
                documents=[chunk["content"]],
                embeddings=[embedding],
                metadatas=[chunk["metadata"]],
                ids=[chunk["metadata"]["chunk_id"]]
            )
            print(f"[RAG] [VectorStore] Added chunk {chunk['metadata'].get('chunk_id', idx)} successfully.")
        except Exception as e:
            print(f"[RAG] Failed to add chunk {chunk['metadata'].get('chunk_id', idx)}: {e}")
        time.sleep(0.2)  # Small delay to reduce rate limit risk
    elapsed = time.time() - start_time
    print(f"[RAG] Embedding and vector store creation completed in {elapsed:.2f} seconds.")
    vector_store = Chroma(
        client=client,
        collection_name=Config.collection_name,
        embedding_function=embeddings
    )
    return documents, all_chunks, vector_store

def load_existing_vector_store(persist_directory: str, embedding_model_name: str, collection_name: str) -> Chroma:
    """Load existing Chroma vector store."""
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        embeddings = GeminiEmbeddings(embedding_model_name)
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"❌ Error loading vector store: {str(e)}")
        return None

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
    """Intelligently merge overlapping and related chunks with temporal awareness."""
    
    @staticmethod
    def detect_table_content(text: str) -> bool:
        """Detect if content contains tabular data."""
        table_indicators = [
            r'\b\d+\s+\w+.*\d+%',  # Pattern like "1 Company Name ... 25%"
            r'Top\s+\d+\s+.*Accounts',  # "Top 20 ... Accounts"
            r'\d+\.\d+\s+\d+\.\d+',  # Multiple decimal numbers
            r'\b\d+%\s+\d+%',  # Multiple percentages
            r'Growth.*%.*\n.*\d+%',  # Growth percentage patterns
            r'\|.*\|',  # Markdown table syntax
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def calculate_overlap_score(chunk1: str, chunk2: str) -> float:
        """Calculate content overlap between two chunks."""
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @classmethod
    def group_related_chunks(cls, chunks: List[Dict], month: Optional[str] = None, quarter: Optional[str] = None) -> List[List[Dict]]:
        """Group chunks by temporal context and content similarity."""
        if not chunks:
            return []
        
        # Filter by temporal context if specified
        filtered_chunks = chunks
        if month:
            filtered_chunks = [c for c in filtered_chunks if c['metadata'].get('month') == month]
        if quarter:
            filtered_chunks = [c for c in filtered_chunks if c['metadata'].get('quarter') == quarter]
        
        # Group by page, source, and month
        page_groups = defaultdict(list)
        for chunk in filtered_chunks:
            page = chunk['metadata'].get('page', 'unknown')
            source = chunk['metadata'].get('source_file', 'unknown')
            month = chunk['metadata'].get('month', 'unknown')
            key = f"{source}_{month}_{page}"
            page_groups[key].append(chunk)
        
        # Within each group, merge by content similarity or table content
        final_groups = []
        
        for page_key, page_chunks in page_groups.items():
            if len(page_chunks) == 1:
                final_groups.append(page_chunks)
                continue
            
            page_chunks.sort(key=lambda x: x['metadata'].get('chunk_id', 0))
            has_table = any(cls.detect_table_content(chunk['content']) for chunk in page_chunks)
            
            if has_table:
                final_groups.append(page_chunks)
            else:
                groups = []
                for chunk in page_chunks:
                    added_to_group = False
                    for group in groups:
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
        
        sorted_chunks = sorted(chunk_group, key=lambda x: x['metadata'].get('chunk_id', 0))
        combined_content = ""
        seen_content = set()
        
        for chunk in sorted_chunks:
            content = chunk['content'].strip()
            if content not in seen_content:
                combined_content += content + "\n\n"
                seen_content.add(content)
        
        merged_chunk = {
            'rank': sorted_chunks[0]['rank'],
            'content': combined_content.strip(),
            'score': min(chunk['score'] for chunk in sorted_chunks),
            'metadata': sorted_chunks[0]['metadata'].copy(),
            'source_file': sorted_chunks[0]['source_file'],
            'page': sorted_chunks[0]['page'],
            'month': sorted_chunks[0]['metadata'].get('month', 'unknown'),
            'quarter': sorted_chunks[0]['metadata'].get('quarter', 'unknown'),
            'chunk_id': f"merged_{sorted_chunks[0]['chunk_id']}_to_{sorted_chunks[-1]['chunk_id']}",
            'merged_from': len(sorted_chunks)
        }
        
        return merged_chunk

class CFAAgent:
    """Chartered Financial Analyst Agent for deep financial analysis."""
    
    def __init__(self, genai_client, generation_model: str):
        self.genai_client = genai_client
        self.generation_model = generation_model
    
    def analyze_financial_query(self, query: str, chunks: List[Dict], max_retries: int = 3) -> Dict:
        """Perform deep financial analysis with reasoning steps."""
        reasoning_steps = []
        
        # Step 1: Parse query for temporal and business unit context
        reasoning_steps.append("🔍 Parsing query for temporal and business unit context...")
        month, quarter = self._extract_temporal_context(query)
        business_units = self._extract_business_units(query)
        
        # Step 2: Filter and organize chunks
        relevant_chunks = [c for c in chunks if (
            (not month or c['metadata'].get('month') == month) and
            (not quarter or c['metadata'].get('quarter') == quarter)
        )]
        reasoning_steps.append(f"📊 Identified {len(relevant_chunks)} relevant chunks for {month or 'all months'} and {quarter or 'all quarters'}")
        
        # Step 3: Analyze financial metrics
        analysis = []
        for unit in business_units:
            unit_analysis = f"📈 Analyzing {unit}..."
            reasoning_steps.append(unit_analysis)
            
            # Retrieve metrics: revenue, EBITDA, costs, top accounts, NRR/GRR, monetization
            unit_chunks = [c for c in relevant_chunks if unit.lower() in c['content'].lower()]
            revenue, ebitda, costs = self._extract_financial_metrics(unit_chunks, unit)
            top_accounts = self._extract_top_accounts(unit_chunks)
            nrr_grr = self._extract_nrr_grr(unit_chunks)
            monetization = self._extract_monetization(unit_chunks)
            
            # Perform reasoning
            reasoning_steps.append(f"📊 {unit} Metrics: Revenue={revenue}, EBITDA={ebitda}, Costs={costs}")
            reasoning_steps.append(f"👥 Top Accounts: {top_accounts}")
            reasoning_steps.append(f"📈 NRR/GRR: {nrr_grr}")
            reasoning_steps.append(f"💰 Monetization: {monetization}")
            
            # Root cause analysis
            if "compare" in query.lower() and "ebitda" in query.lower():
                reasoning_steps.append(f"🔍 Comparing EBITDA for {unit}...")
                ebitda_analysis = self._compare_ebitda(unit_chunks, unit, quarter or month)
                analysis.append(ebitda_analysis)
            else:
                analysis.append(f"{unit} Analysis: {revenue}, {ebitda}, {costs}, {top_accounts}, {nrr_grr}, {monetization}")
        
        # Step 4: Generate final response
        prompt = f"""You are a Chartered Financial Analyst (CFA) analyzing RateGain's financials.
        
        Reasoning Steps:
        {chr(10).join(reasoning_steps)}
        
        Context:
        {chr(10).join([f"[Source: {c['source_file']}, Page: {c['page']}, Month: {c['metadata'].get('month')}]n{c['content']}n" for c in relevant_chunks])}
        
        Query: {query}
        
        Instructions:
        - Provide a detailed, fact-based analysis addressing all parts of the query
        - Use the reasoning steps to guide the response
        - Include specific numerical values, sources, and month/quarter context
        - For comparisons, identify root causes (e.g., revenue drivers, cost increases)
        - Ensure factual accuracy by cross-referencing all data
        
        Response:"""
        
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config={'temperature': 0.1, 'top_p': 0.8, 'max_output_tokens': 4000}
                )
                if response.candidates and len(response.candidates) > 0:
                    return {
                        'answer': response.candidates[0].content.parts[0].text,
                        'reasoning_steps': reasoning_steps
                    }
                return {'answer': "❌ No response generated", 'reasoning_steps': reasoning_steps}
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    delay = 2 ** attempt
                    reasoning_steps.append(f"⏳ Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                else:
                    reasoning_steps.append(f"❌ Error generating response: {str(e)}")
                    return {'answer': f"❌ Error: {str(e)}", 'reasoning_steps': reasoning_steps}
        
        return {'answer': "❌ Failed to generate response", 'reasoning_steps': reasoning_steps}
    
    def _extract_temporal_context(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract month and quarter from query."""
        month = None
        quarter = None
        if "latest month" in query.lower():
            month = "December"  # Assuming December 2024 is the latest
        elif "last month" in query.lower():
            month = "November"  # One month before December
        else:
            months = ["April", "May", "June", "July", "August", "September", "October", "November", "December", "March"]
            for m in months:
                if m.lower() in query.lower():
                    month = m
                    break
        if "Q1" in query:
            quarter = "Q1"
        elif "Q2" in query:
            quarter = "Q2"
        elif "Q3" in query:
            quarter = "Q3"
        elif "Q4" in query:
            quarter = "Q4"
        return month, quarter
    
    def _extract_business_units(self, query: str) -> List[str]:
        """Extract business units from query."""
        units = ["Travel BI", "Hospi BI", "Enterprise Connectivity", "Channel Manager", "Uno", "BCV", "MHS", "Adara"]
        return [unit for unit in units if unit.lower() in query.lower()] or units  # Default to all units if none specified
    
    def _extract_financial_metrics(self, chunks: List[Dict], unit: str) -> Tuple[str, str, str]:
        """Extract revenue, EBITDA, and costs for a business unit."""
        revenue = ebitda = costs = "Not found"
        for chunk in chunks:
            if "revenue" in chunk['content'].lower() and unit.lower() in chunk['content'].lower():
                revenue_match = re.search(r"\$\s*[\d,.]+(?:\s*mn)?", chunk['content'])
                if revenue_match:
                    revenue = revenue_match.group(0)
            if "ebitda" in chunk['content'].lower() and unit.lower() in chunk['content'].lower():
                ebitda_match = re.search(r"\$\s*[\d,.]+(?:\s*mn)?|\$-[\d,.]+(?:\s*mn)?", chunk['content'])
                if ebitda_match:
                    ebitda = ebitda_match.group(0)
            if "cost" in chunk['content'].lower() and unit.lower() in chunk['content'].lower():
                cost_match = re.search(r"\$\s*[\d,.]+(?:\s*mn)?", chunk['content'])
                if cost_match:
                    costs = cost_match.group(0)
        return revenue, ebitda, costs
    
    def _extract_top_accounts(self, chunks: List[Dict]) -> str:
        """Extract top accounts information."""
        for chunk in chunks:
            if "top" in chunk['content'].lower() and "account" in chunk['content'].lower():
                return chunk['content'][:200] + "..."  # Truncate for brevity
        return "Not found"
    
    def _extract_nrr_grr(self, chunks: List[Dict]) -> str:
        """Extract NRR/GRR information."""
        for chunk in chunks:
            if "NRR" in chunk['content'] or "GRR" in chunk['content']:
                return chunk['content'][:200] + "..."  # Truncate for brevity
        return "Not found"
    
    def _extract_monetization(self, chunks: List[Dict]) -> str:
        """Extract monetization information."""
        for chunk in chunks:
            if "monetization" in chunk['content'].lower():
                return chunk['content'][:200] + "..."  # Truncate for brevity
        return "Not found"
    
    def _compare_ebitda(self, chunks: List[Dict], unit: str, period: str) -> str:
        """Compare EBITDA across periods."""
        # Simplified comparison logic (extend with actual data extraction)
        return f"Comparing {unit} EBITDA for {period}: Data extraction in progress..."

class RobustRAGSystem:
    """RAG system with semantic chunking, temporal encoding, and CFA Agent."""
    
    def __init__(self, 
                 vector_store_path: str = "data/vector_store",
                 collection_name: str = "pdf_documents",
                 embedding_model: str = "gemini-embedding-exp-03-07",
                 generation_model: str = "gemini-2.5-pro"):
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.embeddings = None
        self.vector_store = None
        self.genai_client = None
        self.chunk_merger = SmartChunkMerger()
        self.cfa_agent = None
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
        """Initialize RAG system with CFA Agent."""
        print("🚀 Initializing Robust RAG System with CFA Agent...")
        
        if not self._check_vector_store_exists():
            print("📋 Vector store not found or empty. Running data processing pipeline...")
            # Simulate loading documents (replace with actual document loading)
            documents = [
                {"filename": "RateGain MIS - July'24 FY 2024-25.pdf", "pages": [{"content": "Sample content..."}]}
                # Add other monthly documents here
            ]
            _, _, vector_store = run_data_processing(documents)
            self.vector_store = vector_store
        else:
            print("✅ Vector store found. Loading existing data...")
            self.vector_store = load_existing_vector_store(
                persist_directory=self.vector_store_path,
                embedding_model_name=self.embedding_model,
                collection_name=self.collection_name
            )
        
        try:
            self.embeddings = GeminiEmbeddings(self.embedding_model)
            print(f"✅ Embedding model loaded: {self.embedding_model}")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {str(e)}")
            return
        
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.genai_client = genai.Client(api_key=api_key)
            self.cfa_agent = CFAAgent(self.genai_client, self.generation_model)
            print(f"✅ Generation model initialized: {self.generation_model}")
        except Exception as e:
            print(f"❌ Failed to initialize generation model: {str(e)}")
            return
        
        print("✅ Robust RAG System ready with CFA Agent!")
    
    def _is_analytical_query(self, query: str) -> bool:
        """Determine if a query requires deep financial analysis."""
        analytical_keywords = ["compare", "analyze", "why", "reason", "trend", "increase", "decrease", "root cause"]
        return any(keyword in query.lower() for keyword in analytical_keywords)
    
    def retrieve_and_merge_chunks(self, query: str, k: int = 15, month: Optional[str] = None, quarter: Optional[str] = None) -> List[Dict]:
        """Retrieve and merge chunks with temporal filtering."""
        if not self.vector_store:
            print("❌ Vector store not available")
            return []
        
        try:
            print(f"🔍 Searching for relevant chunks (retrieving {k})...")
            results = self.vector_store.similarity_search_with_score(query, k=k)
            raw_chunks = []
            for i, (doc, score) in enumerate(results):
                chunk_info = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'score': score,
                    'metadata': doc.metadata,
                    'source_file': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'month': doc.metadata.get('month', 'Unknown'),
                    'quarter': doc.metadata.get('quarter', 'Unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
                }
                raw_chunks.append(chunk_info)
            
            print(f"📊 Retrieved {len(raw_chunks)} raw chunks")
            chunk_groups = self.chunk_merger.group_related_chunks(raw_chunks, month, quarter)
            merged_chunks = [self.chunk_merger.merge_chunk_group(group) for group in chunk_groups]
            merged_chunks.sort(key=lambda x: x['score'])
            
            print(f"✅ Merged into {len(merged_chunks)} intelligent chunks")
            for chunk in merged_chunks:
                if 'merged_from' in chunk:
                    print(f"   📋 Merged chunk from {chunk['merged_from']} original chunks (Page: {chunk['page']}, Month: {chunk['month']})")
            
            return merged_chunks
        except Exception as e:
            print(f"❌ Error during retrieval and merging: {str(e)}")
            return []
    
    def generate_response(self, query: str, merged_chunks: List[Dict], max_retries: int = 3) -> str:
        """Generate response for direct queries."""
        if not self.genai_client:
            return "❌ Generation model not available"
        if not merged_chunks:
            return "❌ No relevant information found to answer your query."
        
        context_parts = []
        for chunk in merged_chunks:
            source_info = f"Source: {chunk['source_file']}, Page: {chunk['page']}, Month: {chunk['month']}"
            if 'merged_from' in chunk:
                source_info += f" (Merged from {chunk['merged_from']} chunks)"
            context_parts.append(f"[{source_info}]\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        prompt = f"""You are a helpful AI assistant answering financial queries based on RateGain's documents.

        Context:
        {context}

        Query: {query}

        Instructions:
        - Provide a precise, fact-based answer using only the provided context
        - Include specific numerical values and source references
        - Ensure temporal accuracy (e.g., month, quarter)
        - Do not infer or generate data beyond the context

        Answer:"""
        
        for attempt in range(max_retries):
            try:
                response = self.genai_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config={'temperature': 0.1, 'top_p': 0.8, 'max_output_tokens': 2000}
                )
                if response.candidates and len(response.candidates) > 0:
                    return response.candidates[0].content.parts[0].text
                return "❌ No response generated"
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    delay = 2 ** attempt
                    print(f"⏳ Rate limit hit. Waiting {delay}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                else:
                    print(f"❌ Error generating response: {str(e)}")
                    return f"❌ Error: {str(e)}"
        return "❌ Failed to generate response"
    
    def query(self, question: str, k: int = 15) -> Dict:
        """Handle single or multiple queries with temporal context."""
        print("=" * 80)
        print(f"🔍 ROBUST RAG QUERY: {question}")
        print("=" * 80)
        
        # Split query if it contains multiple questions
        sub_queries = question.split(";") if ";" in question else [question]
        results = []
        
        for sub_query in sub_queries:
            sub_query = sub_query.strip()
            if not sub_query:
                continue
                
            # Extract temporal context
            month, quarter = self.cfa_agent._extract_temporal_context(sub_query)
            
            # Determine if analytical or direct query
            if self._is_analytical_query(sub_query):
                print(f"🧠 Routing to CFA Agent for analytical query: {sub_query}")
                k_for_analytical = 30  # Retrieve more chunks for analysis
                merged_chunks = self.retrieve_and_merge_chunks(sub_query, k=k_for_analytical, month=month, quarter=quarter)
                if not merged_chunks:
                    results.append({
                        'query': sub_query,
                        'answer': "❌ No relevant information found.",
                        'reasoning_steps': [],
                        'sources': [],
                        'chunks_retrieved': 0
                    })
                    continue
                
                cfa_result = self.cfa_agent.analyze_financial_query(sub_query, merged_chunks)
                answer = cfa_result['answer']
                reasoning_steps = cfa_result['reasoning_steps']
            else:
                print(f"📜 Processing direct query: {sub_query}")
                merged_chunks = self.retrieve_and_merge_chunks(sub_query, k=k, month=month, quarter=quarter)
                if not merged_chunks:
                    results.append({
                        'query': sub_query,
                        'answer': "❌ No relevant information found.",
                        'reasoning_steps': [],
                        'sources': [],
                        'chunks_retrieved': 0
                    })
                    continue
                
                answer = self.generate_response(sub_query, merged_chunks)
                reasoning_steps = []
            
            sources = []
            for chunk in merged_chunks:
                source_info = {
                    'file': chunk['source_file'],
                    'page': chunk['page'],
                    'month': chunk['month'],
                    'score': chunk['score'],
                    'merged_from': chunk.get('merged_from', 1)
                }
                if source_info not in sources:
                    sources.append(source_info)
            
            results.append({
                'query': sub_query,
                'answer': answer,
                'reasoning_steps': reasoning_steps,
                'sources': sources,
                'chunks_retrieved': len(merged_chunks)
            })
        
        # Combine results for output
        combined_answer = "\n\n".join([f"Query: {r['query']}\nAnswer: {r['answer']}" for r in results])
        combined_sources = []
        for r in results:
            combined_sources.extend(r['sources'])
        combined_reasoning = []
        for r in results:
            combined_reasoning.extend(r['reasoning_steps'])
        
        print(f"\n📝 FINAL ANSWER:")
        print(combined_answer)
        print(f"\n📚 SOURCES (Merged):")
        for source in combined_sources:
            merge_info = f" (Merged from {source['merged_from']} chunks)" if source['merged_from'] > 1 else ""
            print(f"- {source['file']}, Page: {source['page']}, Month: {source['month']} (Score: {source['score']:.4f}){merge_info}")
        if combined_reasoning:
            print("\n🧠 REASONING STEPS:")
            for step in combined_reasoning:
                print(f"- {step}")
        
        print("=" * 80)
        return {
            'query': question,
            'answer': combined_answer,
            'reasoning_steps': combined_reasoning,
            'sources': combined_sources,
            'chunks_retrieved': sum(r['chunks_retrieved'] for r in results)
        }
    
    def is_ready(self) -> bool:
        """Check if the robust RAG system is ready."""
        return all([
            self.embeddings is not None,
            self.vector_store is not None,
            self.genai_client is not None,
            self.cfa_agent is not None
        ])

def main():
    """Test the robust RAG system with CFA Agent."""
    rag_system = RobustRAGSystem()
    
    if not rag_system.is_ready():
        print("❌ Robust RAG system initialization failed!")
        return
    
    test_queries = [
        "What was the GAAP revenue for Hospi BI in July 2024?",
        "Compare the EBITDA for Hospi BI and Travel BI in Q2 and Q3; analyze why there has been any increase or decrease",
        "What are all the growth percentages in the top 20 demand booster accounts for July 2024?"
    ]
    
    print("\n🧪 Testing Robust RAG System...")
    for query in test_queries:
        result = rag_system.query(query, k=15)
        print(f"\n{'='*20} NEXT QUERY {'='*20}")
        time.sleep(1)

if __name__ == "__main__":
    main()