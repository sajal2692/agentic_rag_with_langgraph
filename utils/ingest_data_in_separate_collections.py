#!/usr/bin/env python3
"""
Script to ingest data files into separate ChromaDB collections using LangChain.
Each CSV file gets its own collection, with each row becoming a document.
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

def get_collection_name(file_path: str) -> str:
    """Generate collection name from file path."""
    filename = Path(file_path).stem
    # Replace any characters that might cause issues in collection names
    collection_name = filename.replace(" ", "_").replace("-", "_").lower()
    return collection_name

def main():
    """Main function to ingest each data file into its own ChromaDB collection."""
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=openai_api_key
    )

    # Setup ChromaDB directory
    chroma_db_path = Path(__file__).parent.parent / "vector_store" / "chroma_db_separate"
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    # Get all CSV files from data directory
    data_dir = Path(__file__).parent.parent / "data"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data directory")
        return

    print(f"Found {len(csv_files)} CSV files to process into separate collections")

    # Process each CSV file into its own collection
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Generate collection name from filename
            collection_name = get_collection_name(str(csv_file))

            print(f"\nProcessing {csv_file.name} -> collection: {collection_name}")

            # Load documents from CSV
            documents = load_csv_as_documents(str(csv_file))

            if not documents:
                print(f"  No documents found in {csv_file.name}")
                continue

            # Create vector store for this collection
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(chroma_db_path)
            )

            # Add documents to the collection in batches
            batch_size = 100
            total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)

            print(f"  Ingesting {len(documents)} documents into collection '{collection_name}'...")

            for i in tqdm(range(0, len(documents), batch_size), desc=f"Ingesting {collection_name}", total=total_batches):
                batch = documents[i:i+batch_size]
                vector_store.add_documents(batch)

            print(f"  Successfully ingested {len(documents)} documents into collection '{collection_name}'")

        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    print(f"\nAll collections persisted to: {chroma_db_path}")
    print("Collections created:")
    for csv_file in csv_files:
        collection_name = get_collection_name(str(csv_file))
        print(f"  - {collection_name} (from {csv_file.name})")

if __name__ == "__main__":
    main()
