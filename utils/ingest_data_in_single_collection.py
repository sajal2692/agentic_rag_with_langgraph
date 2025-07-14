#!/usr/bin/env python3
"""
Script to ingest data files into a single ChromaDB collection using LangChain.
Each row in the CSV files becomes a separate document in the collection.
"""

import os
import pandas as pd
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def load_csv_as_documents(file_path: str) -> list[Document]:
    """Load CSV file and convert each row to a Document."""
    df = pd.read_csv(file_path)
    documents = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
        # Convert row to string representation
        content = " | ".join([f"{col}: {val}" for col, val in row.items()])
        
        # Create metadata with source file and row index
        metadata = {
            "source": os.path.basename(file_path),
            "row_index": index,
            "file_path": file_path
        }
        
        # Add individual columns as metadata for better filtering
        for col, val in row.items():
            metadata[col] = val
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def main():
    """Main function to ingest all data files into ChromaDB."""
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=openai_api_key
    )
    
    # Setup ChromaDB
    chroma_db_path = Path(__file__).parent.parent / "vector_store" / "chroma_db_single"
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name="techmart_data",
        embedding_function=embeddings,
        persist_directory=str(chroma_db_path)
    )
    
    # Get all CSV files from data directory
    data_dir = Path(__file__).parent.parent / "data"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_documents = []
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            documents = load_csv_as_documents(str(csv_file))
            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} documents from {csv_file.name}")
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
    
    if not all_documents:
        print("No documents to ingest")
        return
    
    print(f"\nIngesting {len(all_documents)} documents into ChromaDB...")
    
    # Add all documents to the vector store with progress bar
    try:
        batch_size = 100
        total_batches = len(all_documents) // batch_size + (1 if len(all_documents) % batch_size else 0)
        
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Ingesting documents", total=total_batches):
            batch = all_documents[i:i+batch_size]
            vector_store.add_documents(batch)
        
        print(f"Successfully ingested {len(all_documents)} documents")
        print(f"ChromaDB persisted to: {chroma_db_path}")
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        raise

if __name__ == "__main__":
    main()