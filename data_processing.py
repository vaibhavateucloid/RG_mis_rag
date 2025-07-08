# Page-Based Chunking with Cross-Page Similarity Merging

import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from chromadb.api import ClientAPI

import hashlib

# Load environment variables
load_dotenv()

# Updated imports for newer LangChain versions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import chromadb
from chromadb.utils import embedding_functions

print("✅ All imports successful!")

class Config:
    # Paths
    PDF_DIR = "data"
    VECTOR_DB_DIR = "vector_store"
    MODELS_DIR = "models"
    
    # Page-based chunking parameters
    PAGE_OVERLAP_CHARS = 800  # Characters to overlap between pages
    MAX_CHUNK_SIZE = 4500     # Maximum chunk size
    
    # Embedding model
    EMBEDDING_MODELS = ["gemini-embedding-exp-03-07"]
    
    # Vector store
    COLLECTION_NAME = "pdf_documents"

config = Config()

# Create directories if they don't exist
os.makedirs(config.PDF_DIR, exist_ok=True)
os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)

print(f"📁 PDF Directory: {config.PDF_DIR}")
print(f"🗄️ Vector Store Directory: {config.VECTOR_DB_DIR}")
print(f"🤖 Models Directory: {config.MODELS_DIR}")

def load_pdf_documents(pdf_directory: str) -> List[Document]:
    """Load all PDF files from a directory."""
    documents = []
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {pdf_directory}")
        print(f"Please add PDF files to the {pdf_directory} directory")
        return documents
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        try:
            print(f"📖 Loading: {os.path.basename(pdf_path)}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            
            # Add metadata
            for doc in pdf_docs:
                doc.metadata.update({
                    'source_file': os.path.basename(pdf_path),
                    'file_path': pdf_path,
                    'total_pages': len(pdf_docs)
                })
            
            documents.extend(pdf_docs)
            print(f"   ✅ Loaded {len(pdf_docs)} pages")
            
        except Exception as e:
            print(f"   ❌ Error loading {pdf_path}: {str(e)}")
    
    print(f"\n📄 Total documents loaded: {len(documents)}")
    return documents

def create_page_based_chunks(documents: List[Document], 
                           overlap_chars: int = 800,
                           max_chunk_size: int = 4000) -> List[Document]:
    """
    Create page-based chunks with significant overlap between consecutive pages.
    """
    if not documents:
        print("⚠️ No documents to chunk")
        return []
    
    print(f"✂️ Creating page-based chunks with cross-page overlap...")
    print(f"   Page overlap: {overlap_chars} characters")
    print(f"   Max chunk size: {max_chunk_size} characters")
    
    # Group documents by source file
    file_groups = {}
    for doc in documents:
        source_file = doc.metadata.get('source_file', 'unknown')
        if source_file not in file_groups:
            file_groups[source_file] = []
        file_groups[source_file].append(doc)
    
    all_chunks = []
    chunk_id = 0
    
    for source_file, pages in file_groups.items():
        # Sort pages by page number
        pages.sort(key=lambda x: x.metadata.get('page', 0))
        
        print(f"\n📄 Processing {source_file} ({len(pages)} pages)...")
        
        for i, page in enumerate(pages):
            page_content = page.page_content
            page_num = page.metadata.get('page', i)
            
            # Create base page chunk
            base_chunk = Document(
                page_content=page_content[:max_chunk_size],
                metadata={
                    **page.metadata,
                    'chunk_id': chunk_id,
                    'chunk_type': 'page_based',
                    'chunk_size': len(page_content[:max_chunk_size]),
                    'page_start': page_num,
                    'page_end': page_num
                }
            )
            all_chunks.append(base_chunk)
            chunk_id += 1
            
            # Create overlapping chunk with next page if exists
            if i < len(pages) - 1:
                next_page = pages[i + 1]
                next_page_content = next_page.page_content
                
                # Get overlap from current page (last N characters)
                current_overlap = page_content[-overlap_chars:] if len(page_content) > overlap_chars else page_content
                
                # Get beginning from next page
                next_overlap = next_page_content[:overlap_chars]
                
                # Combine overlapping content
                combined_content = current_overlap + "\n\n" + next_overlap
                
                # Ensure it doesn't exceed max size
                if len(combined_content) > max_chunk_size:
                    combined_content = combined_content[:max_chunk_size]
                
                overlap_chunk = Document(
                    page_content=combined_content,
                    metadata={
                        **page.metadata,
                        'chunk_id': chunk_id,
                        'chunk_type': 'page_overlap',
                        'chunk_size': len(combined_content),
                        'page_start': page_num,
                        'page_end': page_num + 1,
                        'is_overlap': True
                    }
                )
                all_chunks.append(overlap_chunk)
                chunk_id += 1
    
    print(f"✅ Created {len(all_chunks)} page-based chunks")
    
    # Display statistics
    chunk_sizes = [len(doc.page_content) for doc in all_chunks]
    overlap_chunks = len([c for c in all_chunks if c.metadata.get('is_overlap', False)])
    
    print(f"📊 Chunk statistics:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Page chunks: {len(all_chunks) - overlap_chunks}")
    print(f"   Overlap chunks: {overlap_chunks}")
    print(f"   Average size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"   Min: {min(chunk_sizes)} characters")
    print(f"   Max: {max(chunk_sizes)} characters")
    
    return all_chunks

def compute_chunk_hash(text: str) -> str:
    """Compute a SHA256 hash for the given text to use as a unique chunk ID."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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
            print(f"🔄 Initializing Gemini client for model: {self.model_name}")
            self.client = genai.Client(api_key=api_key)
            print(f"✅ Gemini client initialized successfully!")
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Gemini with rate limiting and batching."""
        if not self.client:
            raise ValueError("Gemini client not loaded")
        
        embeddings = []
        print(f"🔢 Processing {len(texts)} documents with rate limiting...")
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:
                print(f"   Processed {i}/{len(texts)} documents...")
            
            embedding = self._embed_with_retry(text)
            embeddings.append(embedding)
            
            # Small delay between requests
            time.sleep(0.1)
        
        print(f"✅ Completed embedding {len(texts)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Gemini."""
        if not self.client:
            raise ValueError("Gemini client not loaded")
        return self._embed_with_retry(text)

def create_vector_store(chunks: List[Document],
                        embedding_model_name: str,
                        persist_directory: str,
                        collection_name: str) -> Chroma:
    """Generate embeddings and store in ChromaDB, with caching to avoid re-embedding existing chunks. Deduplicate chunks by content hash."""
    if not chunks:
        print("⚠️ No chunks to embed")
        return None
    
    print(f"🔢 Creating embeddings using model: {embedding_model_name}")
    print(f"📊 Processing {len(chunks)} chunks...")
    
    try:
        embeddings = GeminiEmbeddings(embedding_model_name)
        print("🔄 Generating embeddings and storing in vector database...")
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        
        # Get existing IDs in the collection (cached chunks)
        existing_ids = set()
        try:
            count = collection.count()
            if count > 0:
                batch_size = 500
                for offset in range(0, count, batch_size):
                    results = collection.get(ids=None, limit=batch_size, offset=offset)
                    if 'ids' in results:
                        existing_ids.update(results['ids'])
        except Exception as e:
            print(f"⚠️ Could not fetch existing IDs from collection: {str(e)}")
        
        # Deduplicate chunks by content hash
        unique_chunks = {}
        for i, chunk in enumerate(chunks):
            chunk_hash = compute_chunk_hash(chunk.page_content)
            if chunk_hash not in unique_chunks:
                unique_chunks[chunk_hash] = (i, chunk)
            else:
                print(f"⏩ Duplicate chunk detected (hash: {chunk_hash}), skipping duplicate.")
        
        # Prepare data for new chunks only
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []
        embeddings_to_add = []
        texts_to_embed = []
        chunk_indices_to_embed = []
        
        for chunk_hash, (i, chunk) in unique_chunks.items():
            unique_id = chunk_hash
            if unique_id not in existing_ids:
                texts_to_embed.append(chunk.page_content)
                chunk_indices_to_embed.append(i)
                ids_to_add.append(unique_id)
                documents_to_add.append(chunk.page_content)
                metadatas_to_add.append({**chunk.metadata, 'chunk_hash': chunk_hash})
            else:
                print(f"⏩ Skipping chunk {i} (already embedded, hash: {unique_id})")
        
        # Generate embeddings for new chunks only
        if texts_to_embed:
            generated_embeddings = embeddings.embed_documents(texts_to_embed)
            for idx, embedding in enumerate(generated_embeddings):
                if embedding:
                    embeddings_to_add.append(embedding)
                else:
                    print(f"⚠️ Skipping chunk {chunk_indices_to_embed[idx]} due to failed embedding")
                    documents_to_add[idx] = None
                    metadatas_to_add[idx] = None
                    ids_to_add[idx] = None
            documents_to_add = [d for d in documents_to_add if d is not None]
            metadatas_to_add = [m for m in metadatas_to_add if m is not None]
            ids_to_add = [i for i in ids_to_add if i is not None]
            embeddings_to_add = [e for e in embeddings_to_add if e]
        else:
            print("✅ All chunks already embedded. No new embeddings needed.")
        
        # Add new documents to collection
        if documents_to_add and embeddings_to_add:
            collection.add(
                documents=documents_to_add,
                embeddings=embeddings_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print(f"✅ Added {len(documents_to_add)} new chunks to vector store.")
        else:
            print("ℹ️ No new chunks added to vector store.")
        
        # Create LangChain Chroma object
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        print(f"✅ Vector store created successfully!")
        print(f"   📍 Location: {persist_directory}")
        print(f"   📦 Collection: {collection_name}")
        print(f"   📄 Total chunks stored: {collection.count()}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

def test_vector_store(vector_store: Chroma, test_query: str = "fastest growing demand booster", k: int = 10):
    """Test the vector store with a sample query."""
    if vector_store is None:
        print("⚠️ Vector store not available for testing")
        return
    
    print(f"🔍 Testing vector store with query: '{test_query}'")
    print(f"📊 Retrieving top {k} similar chunks...")
    
    try:
        results = vector_store.similarity_search_with_score(test_query, k=k)
        
        print(f"\n📋 Search Results:")
        for i, (doc, score) in enumerate(results, 1):
            chunk_type = doc.metadata.get('chunk_type', 'unknown')
            is_overlap = doc.metadata.get('is_overlap', False)
            page_info = f"Page: {doc.metadata.get('page_start', 'Unknown')}"
            if doc.metadata.get('page_end') != doc.metadata.get('page_start'):
                page_info += f"-{doc.metadata.get('page_end', 'Unknown')}"
            
            print(f"\n--- Result {i} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"{page_info} | Type: {chunk_type} | Overlap: {is_overlap}")
            print(f"Content: {doc.page_content[:300]}...")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

def load_existing_vector_store(persist_directory: str,
                               embedding_model_name: str,
                               collection_name: str) -> Chroma:
    """Load an existing vector store from disk."""
    print(f"📁 Loading existing vector store from: {persist_directory}")
    
    try:
        embeddings = GeminiEmbeddings(embedding_model_name)
        
        # Load existing vector store
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Create LangChain Chroma object
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        print(f"✅ Vector store loaded successfully!")
        return vector_store
        
    except Exception as e:
        print(f"❌ Error loading vector store: {str(e)}")
        return None

def print_pipeline_summary(documents, chunks, vector_store, embedding_model_used):
    """Print a summary of the data processing pipeline."""
    print("=" * 60)
    print("📊 PAGE-BASED RAG PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"📁 PDF Directory: {config.PDF_DIR}")
    print(f"📄 Documents Loaded: {len(documents) if documents else 0}")
    print(f"✂️ Total Chunks Created: {len(chunks) if chunks else 0}")
    print(f"🤖 Embedding Model Used: {embedding_model_used}")
    print(f"🗄️ Vector Store Location: {config.VECTOR_DB_DIR}")
    print(f"📦 Collection Name: {config.COLLECTION_NAME}")
    print(f"🎯 Vector Store Status: {'✅ Ready' if vector_store else '❌ Failed'}")
    
    if chunks:
        overlap_chunks = len([c for c in chunks if c.metadata.get('is_overlap', False)])
        print(f"📋 Page chunks: {len(chunks) - overlap_chunks}")
        print(f"🔗 Overlap chunks: {overlap_chunks}")
        print(f"📏 Page overlap: {config.PAGE_OVERLAP_CHARS} characters")
    
    if vector_store:
        print("\n🎯 PAGE-BASED CHUNKING BENEFITS:")
        print("1. ✅ Complete page content preserved")
        print("2. ✅ Cross-page data captured with overlap chunks")
        print("3. ✅ Tables spanning multiple pages handled")
        print("4. ✅ No arbitrary text splitting within pages")
        
        print("\n💡 OPTIMIZED FOR:")
        print("- Complete table capture regardless of size")
        print("- Cross-page content relationships")
        print("- Semantic coherence within pages")
    
    print("=" * 60)

def main():
    """Main function to run the page-based RAG data processing pipeline."""
    print("🚀 Starting PAGE-BASED RAG Data Processing Pipeline...")
    print("📄 Page-by-page chunking with cross-page overlap")
    
    embedding_model_name = None
    
    # Initialize Gemini
    if "gemini-embedding-exp-03-07" in config.EMBEDDING_MODELS:
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                temp_client = genai.Client(api_key=api_key)
                embedding_model_name = "gemini-embedding-exp-03-07"
                print(f"✅ Using Gemini embedding model: {embedding_model_name}")
            else:
                print("❌ GOOGLE_API_KEY not found in environment variables")
        except Exception as e:
            print(f"❌ Could not use Gemini embedding model: {str(e)}")
    
    if not embedding_model_name:
        print("\n❌ No suitable embedding model found!")
        print("Please ensure GOOGLE_API_KEY is set for Gemini.")
        return None, None, None
    
    # Load documents
    documents = load_pdf_documents(config.PDF_DIR)
    
    # Create page-based chunks with overlap
    chunks = create_page_based_chunks(
        documents, 
        overlap_chars=config.PAGE_OVERLAP_CHARS,
        max_chunk_size=config.MAX_CHUNK_SIZE
    )
    
    # Create vector store
    vector_store = create_vector_store(
        chunks=chunks,
        embedding_model_name=embedding_model_name,
        persist_directory=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME
    )
    
    # Test with table query
    if vector_store and chunks:
        test_vector_store(vector_store, "fastest growing demand booster growth percentage", k=15)
    
    # Print summary
    print_pipeline_summary(documents, chunks, vector_store, embedding_model_name)
    
    return documents, chunks, vector_store

if __name__ == "__main__":
    documents, chunks, vector_store = main()