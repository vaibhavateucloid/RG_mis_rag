# Multi-Modal Page-Based Chunking with Cross-Page Similarity Merging and Image Processing

import os
import glob
import time
import base64
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from chromadb.api import ClientAPI

import hashlib
import fitz  # PyMuPDF for image extraction
from PIL import Image

# Load environment variables
load_dotenv()

# Updated imports for newer LangChain versions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import chromadb
from chromadb.utils import embedding_functions

print("✅ All imports successful!")

class Config:
    # Paths
    PDF_DIR = "Documents"  # Root folder containing both MIS and IP + Transcript
    MIS_DIR = os.path.join(PDF_DIR, "MIS")
    IP_DIR = os.path.join(PDF_DIR, "IP + Transcript")
    VECTOR_DB_DIR = "data_vector_store"
    MODELS_DIR = "models"
    IMAGES_DIR = "extracted_images"  # Directory for extracted images
    
    # Page-based chunking parameters
    PAGE_OVERLAP_CHARS = 800  # Characters to overlap between pages
    MAX_CHUNK_SIZE = 4500     # Maximum chunk size
    
    # Image processing parameters
    MIN_IMAGE_SIZE = (100, 100)  # Minimum image dimensions to process
    MAX_IMAGE_SIZE = (2048, 2048)  # Maximum image dimensions (resize if larger)
    SUPPORTED_IMAGE_FORMATS = ['PNG', 'JPEG', 'JPG', 'BMP', 'TIFF']
    
    # Embedding model
    EMBEDDING_MODELS = ["gemini-embedding-exp-03-07"]
    
    # Gemini Vision model
    VISION_MODEL = "gemini-2.5-flash"  # Updated to use latest vision model
    
    # Vector store
    COLLECTION_NAME = "pdf_documents"

config = Config()

# Create directories if they don't exist
os.makedirs(config.PDF_DIR, exist_ok=True)
os.makedirs(config.MIS_DIR, exist_ok=True)
os.makedirs(config.IP_DIR, exist_ok=True)
os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.IMAGES_DIR, exist_ok=True)

print(f"📁 PDF Directory: {config.PDF_DIR}")
print(f"📁 MIS Directory: {config.MIS_DIR}")
print(f"📁 IP + Transcript Directory: {config.IP_DIR}")
print(f"🗄️ Vector Store Directory: {config.VECTOR_DB_DIR}")
print(f"🖼️ Images Directory: {config.IMAGES_DIR}")
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

def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract images from PDF with metadata."""
    print(f"🖼️ Extracting images from: {os.path.basename(pdf_path)}")
    
    try:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Filter out images with width < 500 or height < 150
                    if image.size[0] < 500 or image.size[1] < 150:
                        continue
                    
                    # Resize if too large
                    if image.size[0] > config.MAX_IMAGE_SIZE[0] or image.size[1] > config.MAX_IMAGE_SIZE[1]:
                        image.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                    
                    # Save image to disk
                    image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}_img_{img_index}.{image_ext}"
                    image_path = os.path.join(config.IMAGES_DIR, image_filename)
                    image.save(image_path)
                    
                    images.append({
                        'page_num': page_num,
                        'img_index': img_index,
                        'image': image,
                        'image_bytes': image_bytes,
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'image_ext': image_ext,
                        'size': image.size,
                        'source_file': os.path.basename(pdf_path),
                        'bbox': img[1:5] if len(img) > 4 else None
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
        
        doc.close()
        print(f"   ✅ Extracted {len(images)} images from {os.path.basename(pdf_path)}")
        return images
        
    except Exception as e:
        print(f"   ❌ Error extracting images from {pdf_path}: {str(e)}")
        return []

def analyze_image_with_gemini(image: Image.Image, gemini_client, image_metadata: Dict) -> str:
    """Use Gemini Vision to analyze image and extract information."""
    try:
        # Convert PIL Image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Comprehensive analysis prompt
        analysis_prompt = """Analyze this image thoroughly and provide:

1. **Image Type**: Identify if this is a chart, graph, table, diagram, infographic, or other type of visualization.

2. **Detailed Description**: Provide a comprehensive description of what's shown in the image.

3. **Data Extraction**: If this contains numerical data, charts, or tables:
   - Extract all visible numbers, percentages, and values
   - Identify trends, patterns, or key insights
   - List any headers, labels, or categories

4. **Text Content**: Extract all text visible in the image, including:
   - Titles and headings
   - Labels and legends
   - Data values and units
   - Any other readable text

5. **Visual Elements**: Describe colors, symbols, layout, and visual hierarchy.

6. **Business Context**: If this appears to be a business/financial document, identify:
   - Key performance indicators
   - Time periods
   - Comparative data
   - Growth trends

7. **Relationships**: Describe any relationships between different elements in the image.

Please be thorough and specific in your analysis."""
        
        # Generate content with Gemini Vision
        response = gemini_client.models.generate_content(
            model=config.VISION_MODEL,
            contents=[
                {
                    "parts": [
                        {"text": analysis_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )
        
        return response.text
        
    except Exception as e:
        print(f"   ⚠️ Error analyzing image with Gemini: {str(e)}")
        return f"Error analyzing image: {str(e)}"

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
                    'content_type': 'text',
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
                        'content_type': 'text',
                        'chunk_size': len(combined_content),
                        'page_start': page_num,
                        'page_end': page_num + 1,
                        'is_overlap': True
                    }
                )
                all_chunks.append(overlap_chunk)
                chunk_id += 1
    
    print(f"✅ Created {len(all_chunks)} page-based chunks")
    return all_chunks

def create_image_chunks(pdf_files: List[str], gemini_client) -> List[Document]:
    """Create chunks from extracted images using Gemini Vision analysis, combined with all text from the same page."""
    print(f"🖼️ Processing images from {len(pdf_files)} PDF files...")
    all_image_chunks = []
    chunk_id = 0
    total_images = 0
    for pdf_path in pdf_files:
        # Extract images from PDF
        images_data = extract_images_from_pdf(pdf_path)
        total_images += len(images_data)
        if not images_data:
            continue
        # Load all page texts for this PDF (using PyPDFLoader)
        try:
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
        except Exception as e:
            print(f"   ❌ Error loading PDF for page text extraction: {str(e)}")
            pdf_docs = []
        print(f"🔍 Analyzing {len(images_data)} images from {os.path.basename(pdf_path)}...")
        for img_data in images_data:
            try:
                # Analyze image with Gemini Vision
                print(f"   Analyzing image {img_data['img_index']} on page {img_data['page_num']}...")
                image_description = analyze_image_with_gemini(
                    img_data['image'], 
                    gemini_client, 
                    img_data
                )
                # Get full text for the same page
                page_text = ""
                page_num = img_data['page_num']
                if pdf_docs and 0 <= page_num < len(pdf_docs):
                    page_text = pdf_docs[page_num].page_content
                # Combine image analysis and page text
                chunk_content = f"""[IMAGE + PAGE CONTEXT - {img_data['source_file']}]

Image Location: Page {img_data['page_num']}, Image {img_data['img_index']}
Image Size: {img_data['size'][0]}x{img_data['size'][1]} pixels
Image File: {img_data['image_filename']}

--- IMAGE ANALYSIS ---
{image_description}

--- PAGE TEXT ---
{page_text}

---\nThis chunk combines the image analysis and all text from the same page for richer context."""
                # Create combined chunk
                image_chunk = Document(
                    page_content=chunk_content,
                    metadata={
                        'chunk_id': chunk_id,
                        'chunk_type': 'image_plus_page',
                        'content_type': 'image+text',
                        'source_file': img_data['source_file'],
                        'page': img_data['page_num'],
                        'page_start': img_data['page_num'],
                        'page_end': img_data['page_num'],
                        'img_index': img_data['img_index'],
                        'image_path': img_data['image_path'],
                        'image_filename': img_data['image_filename'],
                        'image_width': img_data['size'][0],
                        'image_height': img_data['size'][1],
                        'chunk_size': len(chunk_content),
                        'bbox': str(img_data['bbox']) if img_data['bbox'] is not None else None
                    }
                )
                all_image_chunks.append(image_chunk)
                chunk_id += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"   ❌ Error processing image {img_data['img_index']} on page {img_data['page_num']}: {str(e)}")
                continue
    print(f"✅ Created {len(all_image_chunks)} image+page-context chunks from {total_images} images")
    return all_image_chunks

def compute_chunk_hash(text: str) -> str:
    """Compute a SHA256 hash for the given text to use as a unique chunk ID."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class GeminiEmbeddings(Embeddings):
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
                        collection_name: str) -> Optional[Chroma]:
    """Generate embeddings and store in ChromaDB, with caching to avoid re-embedding existing chunks. Deduplicate chunks by content hash."""
    if not chunks:
        print("⚠️ No chunks to embed")
        return None
    
    # Separate text and image chunks for reporting
    text_chunks = [c for c in chunks if c.metadata.get('content_type') == 'text']
    image_chunks = [c for c in chunks if c.metadata.get('content_type') == 'image']
    
    print(f"🔢 Creating embeddings using model: {embedding_model_name}")
    print(f"📊 Processing {len(chunks)} total chunks ({len(text_chunks)} text, {len(image_chunks)} image)...")
    
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

def test_vector_store(vector_store: Optional[Chroma], test_queries: Optional[List[str]] = None, k: int = 10):
    """Test the vector store with sample queries including image-related ones."""
    if vector_store is None:
        print("⚠️ Vector store not available for testing")
        return
    if not test_queries:
        test_queries = [
            "fastest growing demand booster growth percentage",
            "revenue growth charts and trends",
            "EBITDA margins visualization",
            "employee headcount data",
            "financial performance charts"
        ]
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print(f"📊 Retrieving top {k} similar chunks...")
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            print(f"\n📋 Search Results for '{query}':")
            for i, (doc, score) in enumerate(results[:5], 1):  # Show top 5
                chunk_type = doc.metadata.get('chunk_type', 'unknown')
                content_type = doc.metadata.get('content_type', 'unknown')
                is_overlap = doc.metadata.get('is_overlap', False)
                page_info = f"Page: {doc.metadata.get('page_start', 'Unknown')}"
                if doc.metadata.get('page_end') != doc.metadata.get('page_start'):
                    page_info += f"-{doc.metadata.get('page_end', 'Unknown')}"
                print(f"\n--- Result {i} (Score: {score:.4f}) ---")
                print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"{page_info} | Type: {chunk_type} | Content: {content_type} | Overlap: {is_overlap}")
                if content_type == 'image':
                    print(f"Image: {doc.metadata.get('image_filename', 'Unknown')}")
                print(f"Content: {doc.page_content[:300]}...")
        except Exception as e:
            print(f"❌ Error during testing query '{query}': {str(e)}")

def load_existing_vector_store(persist_directory: str,
                               embedding_model_name: str,
                               collection_name: str) -> Optional[Chroma]:
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

def print_pipeline_summary(documents, text_chunks, image_chunks, vector_store, embedding_model_used):
    """Print a summary of the multi-modal data processing pipeline."""
    print("=" * 80)
    print("📊 MULTI-MODAL PAGE-BASED RAG PIPELINE SUMMARY")
    print("=" * 80)
    
    print(f"📁 PDF Directory: {config.PDF_DIR}")
    print(f"📄 Documents Loaded: {len(documents) if documents else 0}")
    print(f"✂️ Text Chunks Created: {len(text_chunks) if text_chunks else 0}")
    print(f"🖼️ Image Chunks Created: {len(image_chunks) if image_chunks else 0}")
    print(f"🔢 Total Chunks: {(len(text_chunks) + len(image_chunks)) if text_chunks and image_chunks else 0}")
    print(f"🤖 Embedding Model Used: {embedding_model_used}")
    print(f"👁️ Vision Model Used: {config.VISION_MODEL}")
    print(f"🗄️ Vector Store Location: {config.VECTOR_DB_DIR}")
    print(f"📦 Collection Name: {config.COLLECTION_NAME}")
    print(f"🎯 Vector Store Status: {'✅ Ready' if vector_store else '❌ Failed'}")
    
    if text_chunks:
        overlap_chunks = len([c for c in text_chunks if c.metadata.get('is_overlap', False)])
        print(f"📋 Text - Page chunks: {len(text_chunks) - overlap_chunks}")
        print(f"🔗 Text - Overlap chunks: {overlap_chunks}")
        print(f"📏 Page overlap: {config.PAGE_OVERLAP_CHARS} characters")
    
    if image_chunks:
        print(f"🖼️ Image chunks: {len(image_chunks)}")
        print(f"📁 Images stored in: {config.IMAGES_DIR}")
    
    if vector_store:
        print("\n🎯 MULTI-MODAL CHUNKING BENEFITS:")
        print("1. ✅ Complete page content preserved")
        print("2. ✅ Cross-page data captured with overlap chunks")
        print("3. ✅ Visual content (charts, graphs, tables) analyzed and searchable")
        print("4. ✅ Images linked to their source pages")
        print("5. ✅ Multi-modal search across text and visual content")
        print("6. ✅ AI-powered image analysis with detailed descriptions")
        
        print("\n💡 OPTIMIZED FOR:")
        print("- Complete table and chart data capture")
        print("- Cross-page content relationships")
        print("- Visual data extraction and analysis")
        print("- Multi-modal question answering")
        print("- Semantic search across text and images")
    
    print("=" * 80)

def main():
    """Main function to run the multi-modal page-based RAG data processing pipeline."""
    print("🚀 Starting MULTI-MODAL PAGE-BASED RAG Data Processing Pipeline...")
    print("📄 Page-by-page chunking with cross-page overlap + Image Analysis")
    
    embedding_model_name = None
    
    # Initialize Gemini for embeddings
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
        return None, None, None, None
    
    # Initialize Gemini Vision client
    try:
        gemini_vision_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        print(f"✅ Gemini Vision client initialized for model: {config.VISION_MODEL}")
    except Exception as e:
        print(f"❌ Error initializing Gemini Vision client: {str(e)}")
        return None, None, None, None
    
    # Gather all PDFs for text chunking (MIS + IP)
    mis_pdfs = glob.glob(os.path.join(config.MIS_DIR, "*.pdf"))
    ip_pdfs = glob.glob(os.path.join(config.IP_DIR, "*.pdf"))
    all_pdfs = mis_pdfs + ip_pdfs
    print(f"📚 Found {len(all_pdfs)} PDF files for text chunking.")
    # Load documents from each directory only once
    documents = []
    if mis_pdfs:
        documents.extend(load_pdf_documents(config.MIS_DIR))
    if ip_pdfs:
        documents.extend(load_pdf_documents(config.IP_DIR))
    # Create text chunks (all PDFs)
    text_chunks = create_page_based_chunks(
        documents, 
        overlap_chars=config.PAGE_OVERLAP_CHARS,
        max_chunk_size=config.MAX_CHUNK_SIZE
    )
    # Only extract images from MIS PDFs
    print(f"🖼️ Extracting images only from MIS PDFs...")
    image_chunks = create_image_chunks(mis_pdfs, gemini_vision_client)
    # Combine all text and image chunks
    all_chunks = text_chunks + image_chunks
    # Create vector store with both text and image chunks
    vector_store = create_vector_store(
        chunks=all_chunks,
        embedding_model_name=embedding_model_name,
        persist_directory=config.VECTOR_DB_DIR,
        collection_name=config.COLLECTION_NAME
    )
    
    # Test with various queries including image-related ones
    if vector_store and all_chunks:
        test_queries = [
            "fastest growing demand booster growth percentage",
            "revenue growth charts and visualization",
            "EBITDA margins and financial performance",
            "employee headcount trends",
            "business unit performance charts",
            "quarterly financial results"
        ]
        test_vector_store(vector_store, test_queries, k=15)
    
    # Print summary
    print_pipeline_summary(documents, text_chunks, image_chunks, vector_store, embedding_model_name)
    
    return documents, text_chunks, image_chunks, vector_store

if __name__ == "__main__":
    documents, text_chunks, image_chunks, vector_store = main()