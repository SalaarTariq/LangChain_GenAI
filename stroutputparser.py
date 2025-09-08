from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

# Debug: Check if API key is loaded
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
print(f"API Key (first 10 chars): {api_key[:10] if api_key else 'None'}")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=api_key
)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']

)


template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']

)



prompt1 = template1.invoke({'topic':'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result2.content)

