from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Create embedding model instance
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    dimensions=32
)

# Example text to embed
text = "Hello, this is a test sentence for embeddings."

# Generate embeddings
embedding = embeddings.embed_query(text)
print(str(embedding))