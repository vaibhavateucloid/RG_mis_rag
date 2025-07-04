# This is the main entrypoint for the user and will be a streamlit app which will internally call the rag_main.py

from rag_main import RAGSystem

rag = RAGSystem()
result = rag.query("Which is the fastest growing demand booster?")