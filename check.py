import pickle

METADATA_PATH = "./faiss_index_metadata.pkl"  # Adjust path if necessary

try:
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Display number of entries
    print(f"Total Metadata Entries: {len(metadata)}")

    # Print metadata details
    for key, value in metadata.items():
        print(f"ID: {key}")
        print(f"Source: {value.get('source', 'N/A')}")
        print(f"Date: {value.get('date', 'N/A')}")
        print(f"Text Preview: {value.get('text', '')[:100]}...\n")
except Exception as e:
    print(f"Error: {e}")
