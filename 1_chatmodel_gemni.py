from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os



load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature=1)
result = model.invoke("Tell me the history of Pakistan")
print(result.content)