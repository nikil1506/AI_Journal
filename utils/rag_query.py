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
        "2. Every title entry must be fantasy-themed and overexaggerated while the content must be accurate.\n"
        "3. Return the date as it is in the prompt.\n"
        "4. The title must be at most 5 words long.\n"
        "5. The content must be a markdown-formatted brief journal entry with a maximum of ONLY 1 sentence.\n"
        "6. Each entry must be separated by the delimiter: '###ENTRY###'.\n"
        "7. Generate a total of 3 such responses ONLY.\n"
        "Return ONLY the following format:\n\n"
        "###ENTRY###\n"
        "Title: [Fantasy Themed Title]\n"
        f"Date: [{date}]\n"
        "Content:\n"
        "[Markdown journal entry]\n"
        "###ENTRY###\n\n"
        "Here is the context:\n"
        f"{context}\n\n"
    )

    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"].strip()
    response_date=date
    
    return process_journal_response(response_text,response_date)

def process_journal_response(response_text, date):
    """Processes the journal response text into a structured list of JSON objects with mood & emoji analysis."""
    entries = response_text.split(SPECIAL_DELIMITER)
    formatted_entries = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        lines = entry.split("\n")
        title, date, content = None, date, []

        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Date:"):
                date = line.replace("Date:", "").strip()
            else:
                content.append(line)

        if title and date and content:
            full_text = f"{title} {' '.join(content)}"

            sentiment_result = analyze_sentiment(full_text)
            mood_emoji = analyze_emotion(sentiment_result)

            # Ensure mood and emoji are separated correctly
            mood, emoji = mood_emoji.split() if " " in mood_emoji else ("Unknown", "❓")

            formatted_entries.append({
                "title": title,
                "date": date,
                "content": "\n".join(content).strip(),
                "mood": mood,
                "emoji": emoji
            })

    return formatted_entries

def update_cached_rag(query_text):
    """Handles FAISS updates, runs RAG for new entries, and updates the cache with mood & emoji."""
    index, metadata = load_faiss()
    if index is None or metadata is None:
        return {"error": "FAISS database is not initialized."}

    all_dates = get_all_dates(metadata)
    last_cached_date = get_last_cached_date()

    cached_data = load_cached_json()

    if last_cached_date is None:
        # First-time processing: Store all dates and ensure mood & emoji
        for date in all_dates:
            new_data = similarity_search_for_date(query_text, date, metadata)
            if new_data:
                cached_data.extend(new_data)
        if all_dates:
            save_last_cached_date(max(all_dates))
        save_cached_json(add_mood_to_cached_data(cached_data))
        return cached_data

    latest_faiss_date = max(all_dates)

    if last_cached_date == latest_faiss_date:
        cached_data = [entry for entry in cached_data if entry["date"] != latest_faiss_date]
        new_data = similarity_search_for_date(query_text, latest_faiss_date, metadata)
        if new_data:
            cached_data.extend(new_data)
        save_cached_json(add_mood_to_cached_data(cached_data))
        return cached_data

    if last_cached_date < latest_faiss_date:
        new_data = similarity_search_for_date(query_text, latest_faiss_date, metadata)
        if new_data:
            cached_data.extend(new_data)
            save_last_cached_date(latest_faiss_date)
            save_cached_json(add_mood_to_cached_data(cached_data))
        return cached_data

    return cached_data

def add_mood_to_cached_data(cached_data):
    """Ensures every entry in cached data has mood and emoji."""
    for entry in cached_data:
        if "mood" not in entry or "emoji" not in entry:
            sentiment_result = analyze_sentiment(entry["title"] + " " + entry["content"])
            mood_emoji = analyze_emotion(sentiment_result)

            # Ensure mood and emoji are separated
            mood, emoji = mood_emoji.split() if " " in mood_emoji else ("Unknown", "❓")

            entry["mood"] = mood
            entry["emoji"] = emoji
    return cached_data