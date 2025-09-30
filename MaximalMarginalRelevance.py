from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Example docs
docs = [
    Document(page_content="Langchain makes it easy to work with LLMs"),
    Document(page_content="Langchain is used to make LLM based applications"),
    Document(page_content="Chroma is used to store and search document embeddings"),
    Document(page_content="MMR helps you get diverse results when doing similarity search"),
    Document(page_content="Langchain supports Chroma, FAISS, Pinecone and more"),
]

# 🔥 Use free HuggingFace embeddings instead of OpenAI
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vectorstore
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings_model
)

# Retriever with Maximum Marginal Relevance (MMR)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.7}  # balance relevance/diversity
)

# Query
query = "What is LangChain?"
results = retriever.invoke(query)

# Print results
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}")
