from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    template="Write an Essay on the given topic \n {topic}",
    input_variables=["topic"]
)


topic = input("Enter any topic")

fromatted_prompt = prompt.format(topic = topic)

blog_title = llm.invoke(fromatted_prompt)
print("Generated Blog Title: ",blog_title)

