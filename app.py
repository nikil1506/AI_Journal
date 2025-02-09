from flask import Flask, request, jsonify
import faiss
import pickle
import os
from datetime import datetime
import numpy as np
from langchain_ollama import OllamaEmbeddings
from utils import sentiment, rag_query
from utils.vector_store import store_in_vector_db
from utils.events import get_upcoming_events
from utils.motivate import generate_motivation

app = Flask(__name__)

# Define FAISS and storage path
FAISS_DB_PATH = "./faiss_index"
JOURNAL_DIR = "data/journals/"
FAISS_DB_PATH = "faiss_index"
METADATA_PATH = FAISS_DB_PATH + "_metadata.pkl"
embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11666")

# Test an embedding to determine the expected dimension
test_embedding = embeddings.embed_query("Test sentence.")
expected_dim = len(test_embedding)  # Should be 3072

# Load FAISS index and metadata
try:
    if os.path.exists(FAISS_DB_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_DB_PATH)

        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        # Check if the FAISS index dimension matches the embedding model
        if index.d != expected_dim:
            print(f"FAISS index dimension mismatch: expected {expected_dim}, found {index.d}. Deleting old index...")
            os.remove(FAISS_DB_PATH)
            os.remove(METADATA_PATH)
            index = None
            metadata = {}

        print("FAISS vector database loaded successfully.")
    else:
        index = None
        metadata = {}

except Exception as e:
    print(f"Error loading FAISS database: {e}")
    index = None
    metadata = {}

### --- API: Query FAISS --- ###
@app.route('/journal', methods=['POST'])
def run_query_rag():
    """API endpoint to query FAISS-based RAG and retrieve structured journal summaries."""
    
    query_text = "You are my creative writer. Help me write content for my personal journal based on the information I will provide you. You must strictly follow the formatting."

    # Use cached data if FAISS hasn't changed
    response = rag_query.update_cached_rag(query_text)

    return jsonify(response)  # Return JSON directly



### --- API: Analyze Emotion --- ###
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion_api():
    """API endpoint to analyze emotions in a given text."""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text = data["text"]
    result = sentiment.analyze_emotion(text)

    return jsonify({"emotion": result})

### --- API: Recreate FAISS Vector Store --- ###
# @app.route('/recreate_vector_store', methods=['POST'])
# def recreate_vector_store():
#     """API endpoint to reload all Markdown text files from data/journals/ and recreate the FAISS vector store."""
    
#     journal_dir = "data/journals/"
#     text_files = [os.path.join(journal_dir, f) for f in os.listdir(journal_dir) if f.endswith(".md")]

#     total_chunks = 0

#     for file_path in text_files:
#         last_timestamp = get_last_processed_timestamp(file_path)  # Function to track timestamps
        
#         try:
#             file_content = read_markdown(file_path)
#             extracted_text = extract_new_content(file_content, last_timestamp)
#             chunks = split_into_chunks(extracted_text, chunk_size=500, overlap=100)
            
#             if chunks:
#                 store_in_vector_db(chunks, source_file=file_path)
#                 total_chunks += len(chunks)
#                 update_last_processed_timestamp(file_path)  # Update timestamp
#             else:
#                 print(f"No new content found in {file_path}")

#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")

#     return jsonify({
#         "message": "FAISS vector store recreated.",
#         "total_chunks": total_chunks,
#         "processed_files": text_files
#     })
@app.route('/recreate_vector_store', methods=['POST'])
def recreate_vector_store():
    """API endpoint to reload all text files from data/journals/ and recreate the FAISS vector store."""
    
    journal_dir = "data/journals/"
    text_files = [os.path.join(journal_dir, f) for f in os.listdir(journal_dir) if f.endswith(".txt")]

    processed_files = 0

    for file_path in text_files:
        try:
            print(f"Processing {file_path}...")

            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read().strip()

            if not file_content:
                print(f"Skipping {file_path} because it's empty.")
                continue

            # Store entire file as a single chunk
            store_in_vector_db(file_content, source_file=file_path)
            processed_files += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    return jsonify({
        "message": "FAISS vector store recreated.",
        "processed_files": processed_files
    })

@app.route('/send_content', methods=['POST'])
def send_content():
    """API to create/update a journal file based on the provided date (YYYY-MM-DD) and content."""
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid format, expected a JSON object"}), 400

    date_str = data.get("date")
    content = data.get("content")

    if not date_str or not content:
        return jsonify({"error": "Missing required fields: 'date' and 'content'"}), 400

    # Validate date format
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')  # Only process YYYY-MM-DD
        file_name = f"{dt.strftime('%Y-%m-%d')}.txt"
        file_path = os.path.join(JOURNAL_DIR, file_name)
    except ValueError:
        return jsonify({"error": f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD'"}), 400

    # Ensure journal directory exists
    os.makedirs(JOURNAL_DIR, exist_ok=True)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Appending to existing file: {file_path}")
    else:
        print(f"Creating new journal file: {file_path}")

    # Append only the content (No timestamps inside the file)
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"\n{content}\n")

    # Store only the new content in FAISS
    store_in_vector_db(content, source_file=file_path)

    return jsonify({
        "message": "Journal entry saved successfully.",
        "updated_file": file_path
    })

@app.route('/events', methods=['POST'])
def events():
    """API to fetch upcoming events and tasks from the vector store."""
    data = request.get_json()

    if not isinstance(data, dict) or "current_timestamp" not in data:
        return jsonify({"error": "Invalid request format. Expected {'current_timestamp': 'YYYY-MM-DD HH:MM:SS'}"}), 400

    current_timestamp = data["current_timestamp"]

    # Call the function in events.py
    events_list = get_upcoming_events(current_timestamp)

    return jsonify(events_list)

@app.route('/motivate_me', methods=['POST'])
def motivate_me():
    """API endpoint to generate a motivational quote based on achievements."""
    motivation = generate_motivation()
    return jsonify({"text": motivation})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
