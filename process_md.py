import re
from datetime import datetime

def read_markdown(file_path):
    """Reads a markdown file and returns its content as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_new_content(text, last_known_timestamp):
    """
    Extracts content from the last known timestamp onward.
    
    Args:
        text (str): The markdown content.
        last_known_timestamp (str): The last processed timestamp in format '%Y-%m-%d %H:%M:%S'.
    
    Returns:
        Extracted content as a string.
    """
    # Find all timestamps in the text (assuming they are formatted as YYYY-MM-DD HH:MM:SS)
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    timestamps = re.findall(timestamp_pattern, text)

    if not timestamps:
        return ""  # No timestamps found

    # Convert timestamps to datetime objects
    timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
    last_timestamp_dt = datetime.strptime(last_known_timestamp, '%Y-%m-%d %H:%M:%S')

    # Find the first occurrence of a timestamp later than the last known one
    for ts in timestamps:
        if ts > last_timestamp_dt:
            # Extract content from this timestamp onward
            split_text = re.split(timestamp_pattern, text)
            index = split_text.index(ts.strftime('%Y-%m-%d %H:%M:%S'))
            return ''.join(split_text[index:])  # Return content from this timestamp onward

    return ""  # No new content found

def split_into_chunks(text, chunk_size=500, overlap=100):
    """
    Splits text into chunks with overlap.
    
    Args:
        text (str): The text to be chunked.
        chunk_size (int): Number of characters per chunk.
        overlap (int): Number of characters to overlap between chunks.
    
    Returns:
        List of text chunks.
    """
    words = re.split(r'(\s+)', text)  # Split by spaces but keep them
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ''.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward with overlap
    
    return chunks

# Example Usage
file_path = "data/journal.md"  
last_timestamp = "2025-02-08 12:00:00"  

md_content = read_markdown(file_path)
new_content = extract_new_content(md_content, last_timestamp)

if new_content:
    chunks = split_into_chunks(new_content, chunk_size=500, overlap=100)
    
    # Display first few chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}")
else:
    print("No new content found since the last known timestamp.")
