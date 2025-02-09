import json
import numpy as np
from utils.vector_store import load_existing_faiss
import ollama
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from utils.sentiment import analyze_sentiment, analyze_emotion  # Importing sentiment functions

# Load FAISS index and metadata
index, metadata = load_existing_faiss()
query_text = "You are my creative writer. Help me write content for my personal journal based on the information I will provide you. You must strictly follow the formatting."

SPECIAL_DELIMITER = "###ENTRY###"  # Special delimiter for separating journal entries

def similarity_search(query_text=query_text, top_k=3):
    """Performs similarity search and generates structured journal entries using Llama3."""
    if index is None:
        return {"error": "FAISS database is not initialized."}

    # Compute query embedding
    embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11666")
    query_embedding = np.array([embeddings.embed_query(query_text)], dtype=np.float32)

    # Retrieve all stored embeddings from FAISS index
    stored_embeddings = index.reconstruct_n(0, index.ntotal)  # Get all embeddings
    stored_embeddings = np.array(stored_embeddings, dtype=np.float32)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]  # Get similarity scores

    # Get top K most similar documents
    top_k_indices = np.argsort(similarity_scores)[::-1][:top_k]  # Sort descending

    # Retrieve text chunks from FAISS
    retrieved_docs = [metadata[idx]["text"] for idx in top_k_indices if idx in metadata]

    if not retrieved_docs:
        return {"error": "No relevant journal entries found."}

    # Combine retrieved documents into context
    context = "\n\n".join(retrieved_docs)
    prompt = (
        "You are a structured assistant that processes journal entries based on what I talked to you.\n"
        "Take the provided speech-to-text data and organize it into journal entries with a title and content.\n"
        "Follow these strict formatting rules:\n"
        "1. Strictly answer in the first person.\n"
        "2. Every journal entry must be fantasy-themed and overexaggerated.\n"
        "3. Each entry must have a unique date.\n"
        "4. The title must be at most 5 words long.\n"
        "5. The content must be a markdown-formatted brief journal entry with a maximum of 4 to 5 sentences\n"
        "6. Each entry must be separated by the delimiter: '###ENTRY###'.\n\n"
        "Return ONLY the following format:\n\n"
        "###ENTRY###\n"
        "Title: [Fantasy Themed Title]\n"
        "Date: [Date]\n"
        "Content:\n"
        "[Markdown journal entry]\n"
        "###ENTRY###\n\n"
        "Here is the context:\n"
        f"{context}\n\n"
        "Ensure the JSON structure remains intact, contains no explanations, and strictly follows the above constraints.\n"
        "The response must only contain characters allowed in a JSON.\n"
    )

    # Query Llama3 for structured journal entries
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])

    # Extract response text
    response_text = response["message"]["content"].strip()

    # Process the structured response and add mood analysis
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
            sentiment_result = analyze_sentiment(text_for_analysis)  # Get sentiment
            mood_emoji = analyze_emotion(sentiment_result)  # Get mood + emoji

            # Ensure mood and emoji are separated
            mood, emoji = mood_emoji.split()  # Splits into ["Mood", "Emoji"]

            formatted_entries.append({
                "title": title,
                "date": date,
                "content": "\n".join(content).strip(),
                "mood": mood,  # Separate mood
                "emoji": emoji  # Separate emoji
            })

    return formatted_entries
