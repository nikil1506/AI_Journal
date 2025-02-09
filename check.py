from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11666")
test_embedding = embeddings.embed_query("Hello, world!")
print(f"Test embedding dimension: {len(test_embedding)}")  # Should print 3072
