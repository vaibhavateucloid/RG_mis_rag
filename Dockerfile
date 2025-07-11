# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
# RUN apt-get update && apt-get install -y ...

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code, including data_vector_store
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports (adjust as needed)
EXPOSE 8501 8000

# Default command (adjust if you use Streamlit/Chainlit UI)
CMD ["chainlit", "run", "rag_main.py"] 