import faiss
import pickle
import json
import numpy as np
import ollama
from datetime import datetime

# Define FAISS storage paths
FAISS_DB_PATH = "./faiss_index"
METADATA_PATH = FAISS_DB_PATH + "_metadata.pkl"

def load_vector_store():
    """Loads the FAISS index and metadata."""
    try:
        index = faiss.read_index(FAISS_DB_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception:
        print("FAISS database not found or failed to load.")
        return None, {}

def get_upcoming_events(current_timestamp):
    """Uses the LLM to find important upcoming tasks from the vector store."""
    index, metadata = load_vector_store()
    if index is None:
        return {"error": "FAISS database is not initialized."}

    # Convert timestamp to datetime object
    try:
        current_dt = datetime.strptime(current_timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return {"error": "Invalid timestamp format. Expected 'YYYY-MM-DD HH:MM:SS'."}

    # Retrieve all journal entries from metadata
    journal_entries = [meta["text"] for meta in metadata.values()]

    if not journal_entries:
        return {"error": "No journal entries found in the vector store."}

    # Combine entries into a single context
    context = "\n\n".join(journal_entries)

    # Prompt the LLM to extract relevant upcoming tasks
    prompt = (
        "You are a structured assistant that extracts tasks from journal entries. "
        "Given the journal entries below, find ONLY references to upcoming events(Like exams or projects or meetings), deadlines, or tasks. "
        "Upcoming events means any event that has a higher value for date than the current time"
        "Return only a valid JSON list where each object contains:\n"
        " - 'date': The event's date (e.g., 'Feb 5')\n"
        " - 'content': A short description of the task\n\n"
        "Return ONLY a JSON array like this:\n\n"
        "[\n"
        "  {\n"
        '    "date": "Feb 5",\n'
        '    "content": "You need to revise for your exam"\n'
        "  },\n"
        "  {\n"
        '    "date": "Feb 6",\n'
        '    "content": "Submit your project report"\n'
        "  }\n"
        "]\n\n"
        f"Context:\n{context}\n\n"
        "Make sure the response is a valid JSON array with no extra text."
    )

    # Query Llama3 for event extraction
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])

    # Extract and parse LLM response
    response_text = response["message"]["content"].strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output as JSON.", "raw_response": response_text}
