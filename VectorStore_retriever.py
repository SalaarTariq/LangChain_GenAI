from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Example documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="Google provides free open-source embedding models like EmbeddingGemma")
]

# Use a free embedding model
# Option 1: Small & fast SentenceTransformer (good baseline)
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Option 2: Google’s open EmbeddingGemma model
model_name = "google/embeddinggemma-300m"

embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Create a Chroma vector store
vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Run a query
query = "what is chroma used for"
results = retriever.invoke(query)

# Print results
for r in results:
    print(f"Score: {r.metadata if r.metadata else 'N/A'}")
    print(f"Content: {r.page_content}\n")
