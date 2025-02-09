import json
import numpy as np
from utils.vector_store import load_existing_faiss
import ollama
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load FAISS index and metadata
index, metadata = load_existing_faiss()
query_text = ("You are my creative writer. Help me write content for my personal journal based on the information I will provide you. You must strictly follow the formattings")

def similarity_search(query_text=query_text, top_k=3):
    """Performs similarity search instead of FAISS query search and generates structured summaries using Llama3."""
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
        "You are a structured assistant that processes journal entries based on what I talked to you. "
        "Take the provided speech-to-text data and organize them into a content and title. "
        "Strictly answer in first person. "
        "Make sure you provide me content for all the different dates in the journal. "
        "Every new date should have its own set of title and content. "
        "The Title should be at most 5 words. "
        "The content should be a brief description of the happenings of whatever happened on that particular day with hints of my emotions you get from the information. "
        "There must not be any repeated entries for the same date. "
        "Summary and title must always be present. "
        "Each entry should include:\n"
        " - 'title': A short phrase summarizing the day's theme\n"
        " - 'date': The journal entry's date (e.g., 'Feb 5')\n"
        " - 'content': The journal text formatted in markdown\n\n"
        "Return ONLY a valid JSON array, following this format:\n\n"
        "[\n"
        "  {\n"
        '    "title": "Started Exam Prep",\n'
        '    "date": "Feb 5",\n'
        '    "content": "Journal is markdown format"\n'
        "  },\n"
        "  {\n"
        '    "title": "Feeling Sick",\n'
        '    "date": "Feb 6",\n'
        '    "content": "Journal is markdown format"\n'
        "  }\n"
        "]\n\n"
        "Here is the context:\n"
        f"{context}\n\n"
        "Ensure that the JSON structure remains intact and does not contain any explanations or extra text. "
        "Strip all string escape characters in all fields except content. Content should be a proper markdown. "
        "The response must only contain characters allowed in a JSON. "
        "ALL THE ABOVE CONSTRAINTS MUST BE STRICTLY FOLLOWED."
    )

    # Query Llama3 for structured summary
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])

    # Extract response text and clean it
    response_text = response["message"]["content"].strip()

    # Parse response JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse Llama3 output as JSON.", "raw_response": response_text}
