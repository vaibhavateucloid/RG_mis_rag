# This is going to be my main backend file where I will build a rag solution

# Basic steps to build a RAG with Gemini:
# 1. Load Data: Load your documents (e.g., PDFs, text files, web pages).
# 2. Chunk Data: Split the documents into smaller, manageable chunks.
# 3. Embed Chunks: Generate embeddings for each chunk using a Gemini embedding model.
# 4. Store Embeddings: Store the chunks and their embeddings in a vector database.
# 5. User Query: Receive a user query.
# 6. Embed Query: Generate an embedding for the user query.
# 7. Retrieve Relevant Chunks: Use the query embedding to find the most relevant chunks from the vector database.
# 8. Augment Prompt: Combine the retrieved chunks with the user query to create an augmented prompt.
# 9. Generate Response: Send the augmented prompt to a Gemini Pro model to generate a response.

import os
import glob
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

import chromadb
from chromadb.utils import embedding_functions

print("✅ All imports successful!")

class Config:
    # Paths
    PDF_DIR = "data/pdfs"  # Put your PDF files here
    VECTOR_DB_DIR = "data/vector_store"
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, good quality
    
    # Vector store
    COLLECTION_NAME = "pdf_documents"

config = Config()

# Create directories if they don't exist
os.makedirs(config.PDF_DIR, exist_ok=True)
os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

print(f"📁 PDF Directory: {config.PDF_DIR}")
print(f"🗄️ Vector Store Directory: {config.VECTOR_DB_DIR}")

# def main():
#     print("Hello from rg-mis-rag!")


# if __name__ == "__main__":
#     main()
