import json
import os
import ollama
import time
import random
from utils.sentiment import analyze_sentiment

CACHE_FILE = "./cached_rag.json"

motivational_messages = [
    "Keep pushing forward! Every step you take brings you closer to success.",
    "Believe in yourself! You are stronger than you think.",
    "Great things take time. Keep going, and you’ll get there!",
    "Every challenge you overcome makes you stronger and wiser.",
    "Stay positive, work hard, and make it happen!",
    "Difficult roads often lead to beautiful destinations.",
    "Success is the sum of small efforts, repeated day in and day out.",
    "Your only limit is your mind. Stay focused and push through!",
    "Dream big, work hard, stay focused, and surround yourself with good people.",
    "Every setback is a setup for a comeback. Keep moving forward!"
]

def load_cached_rag():
    """Loads the cached RAG responses without modifying them."""
    if not os.path.exists(CACHE_FILE):
        return []
    
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def extract_achievements(cached_data):
    """Extracts achievements or progress-related entries from cached RAG data."""
    achievements = []
    for entry in cached_data:
        sentiment = analyze_sentiment(entry["content"])
        
        # Consider positive & progress-based entries
        if sentiment in ["Proud", "Accomplished", "Successful", "Happy", "Motivated"]:
            achievements.append(entry["content"])
    
    return achievements

def generate_motivation():
    """Generates a motivational quote based on the first found achievement within 10 seconds."""
    cached_data = load_cached_rag()
    start_time = time.time()

    # Find the first achievement within 10 seconds
    for entry in cached_data:
        if time.time() - start_time > 5:
            print("Search timed out. Returning a random motivational message.")
            return random.choice(motivational_messages)

        sentiment = analyze_sentiment(entry["content"])
        if sentiment in ["Proud", "Accomplished", "Successful", "Happy", "Motivated"]:
            context = entry["content"]
            break
    else:
        print("No achievements found. Returning default message.")
        return "Keep pushing forward! Every step you take brings you closer to success."

    prompt = (
        "You are an inspiring life coach. Generate a personalized motivational message based on the user's past achievement.\n"
        "Use the following journal excerpt:\n\n"
        f"{context}\n\n"
        "Your response should be uplifting, personal, and empowering. Keep it short and powerful."
    )

    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"].strip()
