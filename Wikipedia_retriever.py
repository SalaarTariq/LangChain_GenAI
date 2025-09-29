from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2,lang="en")

query = "What is the history of war between pakistan and india in the view of Chinese"

docs = retriever.invoke(query)

print(docs)