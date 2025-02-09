import faiss
import pickle
import numpy as np
import os
import re
from langchain_ollama import OllamaEmbeddings
from datetime import datetime

FAISS_DB_PATH = "faiss_index"
METADATA_PATH = FAISS_DB_PATH + "_metadata.pkl"

def load_existing_faiss():
    """Loads the existing FAISS index and metadata."""
    if os.path.exists(FAISS_DB_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_DB_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, {}

def extract_date_from_filename(filename):
    """Extracts a date from a filename using YYYY-MM-DD format."""
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(date_pattern, filename)
    if match:
        return match.group(1)
    return datetime.now().strftime('%Y-%m-%d')  # Default to current date

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of a defined size with some overlap."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  # Ensure some overlap

    return chunks

def store_in_vector_db(text, source_file="unknown"):
    """Embeds text chunks based on the file date and stores them in FAISS."""
    embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11666")

    try:
        file_date = extract_date_from_filename(source_file)
        chunks = chunk_text(text)  # Split text into chunks

        existing_index, metadata = load_existing_faiss()

        if existing_index is None:
            print("Creating new FAISS index.")
            index = faiss.IndexFlatL2(3072)  # Assuming embedding dimension is 3072
        else:
            index = existing_index

        for chunk in chunks:
            text_embedding = embeddings.embed_query(chunk)
            if not isinstance(text_embedding, list):
                raise ValueError(f"Embedding failed for {source_file}: unexpected type {type(text_embedding)}")

            index.add(np.array([text_embedding], dtype=np.float32))

            metadata[len(metadata)] = {
                "source": source_file,
                "text": chunk,
                "date": file_date
            }

        # Save FAISS index and metadata
        faiss.write_index(index, FAISS_DB_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Stored {len(chunks)} chunks from {source_file} in FAISS with date {file_date}.")

    except Exception as e:
        print(f"Error storing {source_file} in FAISS: {e}")
