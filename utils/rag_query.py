import json
import os
import numpy as np
import faiss
import pickle
import ollama
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from utils.vector_store import load_existing_faiss
from utils.sentiment import analyze_sentiment, analyze_emotion

FAISS_DB_PATH = "./faiss_index"
METADATA_PATH = FAISS_DB_PATH + "_metadata.pkl"
CACHE_FILE = "./cached_rag.json"
LAST_DATE_FILE = "./last_date.txt"
SPECIAL_DELIMITER = "###ENTRY###"

def load_faiss():
    """Loads FAISS index and metadata."""
    try:
        index = faiss.read_index(FAISS_DB_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception:
        print("FAISS database not found or failed to load.")
        return None, {}

def get_all_dates(metadata):
    """Extracts all unique dates from FAISS metadata."""
    return sorted(set(meta["date"] for meta in metadata.values()), reverse=True)

def get_last_cached_date():
    """Reads the last cached date from file."""
    if os.path.exists(LAST_DATE_FILE):
        with open(LAST_DATE_FILE, "r") as f:
            return f.read().strip()
    return None

def save_last_cached_date(last_date):
    """Saves the last processed date to a file."""
    with open(LAST_DATE_FILE, "w") as f:
        f.write(last_date)

def load_cached_json():
    """Loads cached RAG responses from file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []

def save_cached_json(data):
    """Saves updated RAG responses to cache."""
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)

def similarity_search_for_date(query_text, date, metadata):
    """Performs similarity search for a specific date and generates structured journal entries using Llama3."""
    index, metadata = load_faiss()
    if index is None or metadata is None:
        return None

    embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11666")
    query_embedding = np.array([embeddings.embed_query(query_text)], dtype=np.float32)

    stored_embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        stored_embeddings[i] = index.reconstruct(i)

    similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]

    # Filter for entries with the specific date
    matching_indices = [idx for idx in metadata if metadata[idx]["date"] == date]
    top_k_indices = sorted(matching_indices, key=lambda x: similarity_scores[x], reverse=True)

    retrieved_docs = [metadata[idx]["text"] for idx in top_k_indices if idx in metadata]

    if not retrieved_docs:
        return None  # No results for this date

    context = "\n\n".join(retrieved_docs)
    prompt = (
        "You are a structured assistant that processes journal entries.\n"
        "Take the provided speech-to-text data and organize it into journal entries with a title and content.\n"
        "Follow these strict formatting rules:\n"
        "1. Strictly answer in the first person.\n"
        "2. Every title entry must be fantasy-themed and overexaggerated while the journal entry must be meaningful.\n"
        "3. Each entry must have a unique date.\n"
        "4. The title must be at most 5 words long.\n"
        "5. The content must be a markdown-formatted brief journal entry with a maximum of ONLY 1 sentence.\n"
        "6. Each entry must be separated by the delimiter: '###ENTRY###'.\n"
        "7. Date should be STRICTLY in YYYY-MM-DD format.\n"
        "Return ONLY the following format:\n\n"
        "###ENTRY###\n"
        "Title: [Fantasy Themed Title]\n"
        "Date: [Date]\n"
        "Content:\n"
        "[Markdown journal entry]\n"
        "###ENTRY###\n\n"
        "Here is the context:\n"
        f"{context}\n\n"
    )

    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"].strip()
    
    return process_journal_response(response_text)

def process_journal_response(response_text):
    """Processes the journal response text into a structured list of JSON objects and appends mood + emoji analysis."""
    entries = response_text.split(SPECIAL_DELIMITER)
    formatted_entries = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Extract title, date, and content
        lines = entry.split("\n")
        title, date, content = None, None, []
        
        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Date:"):
                date = line.replace("Date:", "").strip()
            else:
                content.append(line)

        if title and date and content:
            text_for_analysis = f"{title}\n" + "\n".join(content)

            # Get Mood + Emoji
            sentiment_result = analyze_sentiment(text_for_analysis)
            mood_emoji = analyze_emotion(sentiment_result)

            # Ensure mood and emoji are separated safely
            mood_emoji_parts = mood_emoji.split()
            if len(mood_emoji_parts) == 2:
                mood, emoji = mood_emoji_parts
            else:
                mood, emoji = "Unknown", "❓"

            formatted_entries.append({
                "title": title,
                "date": date,
                "content": "\n".join(content).strip(),
                "mood": mood,
                "emoji": emoji
            })

    return formatted_entries

def update_cached_rag(query_text):
    """Checks FAISS for new dates, updates RAG cache if necessary, and returns the latest data."""
    index, metadata = load_faiss()
    if index is None or metadata is None:
        return {"error": "FAISS database is not initialized."}

    all_dates = get_all_dates(metadata)
    last_cached_date = get_last_cached_date()

    if last_cached_date and all_dates and last_cached_date >= max(all_dates):
        return load_cached_json()  # No new updates, return cached results

    # Get new dates that need processing
    new_dates = [date for date in all_dates if last_cached_date is None or date > last_cached_date]

    cached_data = load_cached_json()

    for date in new_dates:
        new_data = similarity_search_for_date(query_text, date, metadata)
        if new_data:
            cached_data.extend(new_data)

    if new_dates:
        save_last_cached_date(max(new_dates))  # Update last processed date
        save_cached_json(cached_data)  # Update cached JSON

    return cached_data
