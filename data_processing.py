# RAG Data Processing Pipeline - Offline Version
# Load PDFs → Chunk Text → Generate Embeddings → Store in Vector DB

import os
import glob
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv # Load environment variables from .env file
from google import genai # Correct import for genai.Client()
from chromadb.api import ClientAPI # Import ClientAPI for type hinting

# Load environment variables from .env file
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
    PDF_DIR = "data/pdfs"  # Put your PDF files here
    VECTOR_DB_DIR = "data/vector_store"
    MODELS_DIR = "models"  # Local models directory
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding model priorities (will try in order)
    EMBEDDING_MODELS = [
        "gemini-embedding-exp-03-07"
    ]
    
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
    """
    Load all PDF files from a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files
        
    Returns:
        List of LangChain Document objects
    """
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

def chunk_documents(documents: List[Document],
                   chunk_size: int = 1000,
                   chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked documents
    """
    if not documents:
        print("⚠️ No documents to chunk")
        return []
    
    print(f"✂️ Chunking {len(documents)} documents...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Overlap: {chunk_overlap} characters")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )
    
    # Split documents
    chunked_docs = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunked_docs):
        chunk.metadata.update({
            'chunk_id': i,
            'chunk_size': len(chunk.page_content)
        })
    
    print(f"✅ Created {len(chunked_docs)} chunks")
    
    # Display statistics
    chunk_sizes = [len(doc.page_content) for doc in chunked_docs]
    print(f"📊 Chunk size statistics:")
    print(f"   Average: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"   Min: {min(chunk_sizes)} characters")
    print(f"   Max: {max(chunk_sizes)} characters")
    
    return chunked_docs

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
                # FIXED: Access .embeddings (plural) and get the values from the first embedding
                return result.embeddings[0].values
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        print(f"❌ Max retries reached for text: {text[:50]}...")
                        return []  # Return empty list on final failure
                else:
                    print(f"❌ Error embedding text: {str(e)}")
                    return []  # Return empty list for other errors
        return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Gemini with rate limiting and batching."""
        if not self.client:
            raise ValueError("Gemini client not loaded")
        
        embeddings = []
        print(f"🔢 Processing {len(texts)} documents with rate limiting...")
        
        for i, text in enumerate(texts):
            if i > 0 and i % 10 == 0:  # Progress update every 10 documents
                print(f"   Processed {i}/{len(texts)} documents...")
            
            embedding = self._embed_with_retry(text)
            embeddings.append(embedding)
            
            # Small delay between requests to avoid rate limiting
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
    """
    Generate embeddings and store in ChromaDB using the specified model.
    
    Args:
        chunks: List of chunked documents
        embedding_model_name: Name of the embedding model (local or Gemini)
        persist_directory: Directory to persist the vector store
        collection_name: Name of the collection in ChromaDB
        
    Returns:
        ChromaDB vector store
    """
    if not chunks:
        print("⚠️ No chunks to embed")
        return None
    
    print(f"🔢 Creating embeddings using model: {embedding_model_name}")
    print(f"📊 Processing {len(chunks)} chunks...")
    
    try:
        embeddings = GeminiEmbeddings(embedding_model_name)
        
        print("🔄 Generating embeddings and storing in vector database...")
        
        # Create vector store
        # Extract texts from chunks for batch embedding
        texts_from_chunks = [chunk.page_content for chunk in chunks]

        # Generate embeddings in batch (this will call _embed_with_retry internally for each text)
        # The embed_documents method of GeminiEmbeddings will return a list of lists of floats,
        # where some inner lists might be empty if embedding failed.
        generated_embeddings = embeddings.embed_documents(texts_from_chunks)

        # Prepare data for ChromaDB by filtering out failed embeddings
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []
        embeddings_to_add = []

        for i, embedding in enumerate(generated_embeddings):
            if embedding: # Check if the embedding list is not empty
                chunk = chunks[i] # Get the original chunk
                documents_to_add.append(chunk.page_content)
                metadatas_to_add.append(chunk.metadata)
                # Generate a unique ID for each chunk
                source_file = chunk.metadata.get('source_file', 'unknown_file')
                chunk_id = chunk.metadata.get('chunk_id', i)
                unique_id = f"{source_file}_{chunk_id}"
                ids_to_add.append(unique_id)
                embeddings_to_add.append(embedding)
            else:
                # Log which chunk failed to embed
                print(f"⚠️ Skipping chunk {i} due to failed embedding: {chunks[i].page_content[:50]}...")

        if not documents_to_add:
            print("❌ No successful embeddings generated. Cannot create vector store.")
            return None

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the collection
        # IMPORTANT: When adding pre-computed embeddings, set embedding_function=None
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None # We are providing embeddings directly
        )

        # Add documents and embeddings to the collection
        collection.add(
            documents=documents_to_add,
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        
        # Now, create a LangChain Chroma object from the existing client and collection
        # This allows test_vector_store and load_existing_vector_store to work.
        # Pass the embeddings object here so LangChain can use it for future queries.
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings # Pass the embeddings object for future queries
        )

        print(f"✅ Vector store created successfully!")
        print(f"   📍 Location: {persist_directory}")
        print(f"   📦 Collection: {collection_name}")
        print(f"   📄 Total chunks stored: {len(documents_to_add)}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

def test_vector_store(vector_store: Chroma, test_query: str = "What is this document about?", k: int = 3):
    """
    Test the vector store with a sample query.
    
    Args:
        vector_store: The ChromaDB vector store
        test_query: Query to test with
        k: Number of similar documents to retrieve
    """
    if vector_store is None:
        print("⚠️ Vector store not available for testing")
        return
    
    print(f"🔍 Testing vector store with query: '{test_query}'")
    print(f"📊 Retrieving top {k} similar chunks...")
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(test_query, k=k)
        
        print(f"\n📋 Search Results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
            
        # Test with similarity scores
        print(f"\n🎯 Search Results with Similarity Scores:")
        results_with_scores = vector_store.similarity_search_with_score(test_query, k=k)
        
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n--- Result {i} (Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Content: {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

def load_existing_vector_store(persist_directory: str,
                               embedding_model_name: str,
                               collection_name: str) -> Chroma:
    """
    Load an existing vector store from disk.
    
    Args:
        persist_directory: Directory where vector store is saved
        embedding_model_name: Name of the embedding model (local or Gemini)
        collection_name: Name of the collection
        
    Returns:
        Loaded ChromaDB vector store
    """
    print(f"📁 Loading existing vector store from: {persist_directory}")
    
    try:
        embeddings = GeminiEmbeddings(embedding_model_name)
        
        # Load existing vector store
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Create LangChain Chroma object from the existing client and collection
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings # Pass the embeddings object for future queries
        )
        
        print(f"✅ Vector store loaded successfully!")
        return vector_store
        
    except Exception as e:
        print(f"❌ Error loading vector store: {str(e)}")
        return None

def print_pipeline_summary(documents, chunks, vector_store, embedding_model_used):
    """Print a summary of the data processing pipeline."""
    print("=" * 60)
    print("📊 RAG DATA PROCESSING PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"📁 PDF Directory: {config.PDF_DIR}")
    print(f"📄 Documents Loaded: {len(documents) if documents else 0}")
    print(f"✂️ Total Chunks Created: {len(chunks) if chunks else 0}")
    print(f"🤖 Embedding Model Used: {embedding_model_used}")
    print(f"🗄️ Vector Store Location: {config.VECTOR_DB_DIR}")
    print(f"📦 Collection Name: {config.COLLECTION_NAME}")
    print(f"🎯 Vector Store Status: {'✅ Ready' if vector_store else '❌ Failed'}")
    print(f"🌐 Running Mode: {'☁️ ONLINE (Gemini)' if embedding_model_used.startswith('gemini') else '🔒 OFFLINE'}")
    
    if not documents:
        print("\n⚠️ NO PDF FILES FOUND!")
        print("Please add PDF files to the data/pdfs/ directory")
    
    if vector_store:
        print("\n🎯 NEXT STEPS:")
        print("1. Add more PDF documents to data/pdfs/ directory")
        print("2. Experiment with different chunk sizes and overlap")
        print("3. Build the retrieval and generation components")
        print("4. Create a query interface with Gemini")
        
        print("\n💡 USAGE:")
        print("- Your vector store is ready for RAG queries!")
        print("- Use vector_store.similarity_search(query) to retrieve relevant chunks")
        print("- The vector store persists automatically and can be reloaded")
    
    print("=" * 60)

def main():
    """Main function to run the RAG data processing pipeline."""
    print("🚀 Starting RAG Data Processing Pipeline...")
    
    embedding_model_name = None
    
    # Try to use Gemini first
    if "gemini-embedding-exp-03-07" in config.EMBEDDING_MODELS:
        try:
            # Test if we can initialize Gemini client
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                temp_client = genai.Client(api_key=api_key)
                embedding_model_name = "gemini-embedding-exp-03-07"
                print(f"✅ Using Gemini embedding model: {embedding_model_name}")
            else:
                print("❌ GOOGLE_API_KEY not found in environment variables")
        except Exception as e:
            print(f"❌ Could not use Gemini embedding model: {str(e)}")
            print("Falling back to local models if available.")
    
    # If Gemini is not used or failed, ensure it's the only option
    if not embedding_model_name:
        print("\n❌ No suitable embedding model found!")
        print("Please ensure GOOGLE_API_KEY is set for Gemini.")
        return None, None, None
            
    # Load documents
    documents = load_pdf_documents(config.PDF_DIR)
    
    # Display sample document
    if documents:
        sample_doc = documents[0]
        print(f"\n📋 Sample document preview:")
        print(f"Source: {sample_doc.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {sample_doc.metadata.get('page', 'Unknown')}")
        print(f"Content preview: {sample_doc.page_content[:200]}...")
    
    # Chunk the documents
    chunks = chunk_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    
    # Display sample chunk
    if chunks:
        sample_chunk = chunks[0]
        print(f"\n📋 Sample chunk preview:")
        print(f"Source: {sample_chunk.metadata.get('source_file', 'Unknown')}")
        print(f"Chunk ID: {sample_chunk.metadata.get('chunk_id', 'Unknown')}")
        print(f"Size: {sample_chunk.metadata.get('chunk_size', 'Unknown')} characters")
        print(f"Content: {sample_chunk.page_content[:300]}...")
    
    # Create vector store
    vector_store = create_vector_store(
        chunks=chunks,
        embedding_model_name=embedding_model_name,
        persist_directory=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME
    )
    
    # Test the vector store
    if vector_store and chunks:
        test_vector_store(vector_store)
    
    # Print summary
    print_pipeline_summary(documents, chunks, vector_store, embedding_model_name)
    
    return documents, chunks, vector_store

if __name__ == "__main__":
    documents, chunks, vector_store = main()