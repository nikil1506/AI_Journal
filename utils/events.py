import faiss
import pickle
import json
import numpy as np
import ollama
from datetime import datetime

# Define FAISS storage paths
FAISS_DB_PATH = "./faiss_index"
METADATA_PATH = FAISS_DB_PATH + "_metadata.pkl"
SPECIAL_DELIMITER = "###EVENT###"  # Delimiter for structuring LLM output

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
    """Finds upcoming events from journal metadata and queries the LLM."""
    index, metadata = load_vector_store()
    if index is None:
        return {"error": "FAISS database is not initialized."}

    # Convert timestamp to a comparable date
    try:
    # Convert current timestamp to a datetime object and then extract the date
        current_dt = datetime.strptime(current_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S').date()
    except ValueError:
        return {"error": "Invalid timestamp format. Expected 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS.ssssss'."}


    # Filter metadata for future events only
    upcoming_entries = []
    for meta in metadata.values():
        try:
            entry_date = datetime.strptime(meta["date"], '%Y-%m-%d').date()  # Ensure entry_date is also a date
            if entry_date >= current_dt:  # Now both are date objects, comparison is valid
                upcoming_entries.append(f"{SPECIAL_DELIMITER}\nDate: {entry_date.strftime('%b %d')}\nContent: {meta['text']}")
        except Exception as e:
            print(f"Skipping entry due to date parsing error: {e}")

    if not upcoming_entries:
        return {"error": "No upcoming events found in the vector store."}

    # Combine filtered entries into context
    context = "\n\n".join(upcoming_entries)

    # Your strict event-extraction prompt
    print("prompt Reached")
    prompt = (
        "You are a creative assistant that extracts and organizes upcoming events from journal entries.\n"
        "Given the journal entries below, identify only future events, deadlines, and tasks based on their dates.\n"
        "Date should be in (YYYY MM DD) format STRICTLY"
        "Use the following strict formatting:\n"
        f"1. Each event must be separated by '{SPECIAL_DELIMITER}'.\n"
        "2. Each event must include:\n"
        "   - 'Date: [YYYY MM DD]'\n"
        "   - 'Content: [Short event description]'\n\n"
        "3. The content must be very brief - One sentence of around 8 to 10 words at most and STRICTLY one event per date.\n"
        "4. Events must not include mundane events but only note important events like exams, meetings, deadlines, or reminders to check the progress of new hobbies.\n"
        "Here is the context of journal entries with future events:\n\n"
        f"{context}\n\n"
        "Ensure that the response follows the strict delimiter and date format without any explanations."
    )

    # Query Llama3 for structured event extraction
    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])

    # Extract and parse LLM response
    response_text = response["message"]["content"].strip()

    return process_event_response(response_text)

def process_event_response(response_text):
    """Processes the LLM response into a structured list of JSON objects with date in YYYY-MM-DD format."""
    events = response_text.split(SPECIAL_DELIMITER)
    formatted_events = []

    for event in events:
        event = event.strip()
        if not event:
            continue

        # Extract date and content
        lines = event.split("\n")
        date, content = None, []

        for line in lines:
            if line.startswith("Date:"):
                raw_date = line.replace("Date:", "").strip()
                try:
                    # Convert various date formats to YYYY-MM-DD
                    date_obj = datetime.strptime(raw_date, '%b %d')  # If date is in "Feb 8" format
                    date = datetime(datetime.now().year, date_obj.month, date_obj.day).strftime('%Y-%m-%d')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(raw_date, '%Y-%m-%d')  # If already in YYYY-MM-DD format
                        date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        date = raw_date  # Keep raw date if conversion fails
            elif line.startswith("Content:"):
                content.append(line.replace("Content:", "").strip())
            else:
                content.append(line.strip())

        if date and content:
            formatted_events.append({
                "date": date,
                "content": " ".join(content)  # Keep it in brief format
            })

    return formatted_events