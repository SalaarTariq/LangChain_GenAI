from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
import os

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables= [ "topic"]
)

prompt2= PromptTemplate(
    template='Summarize the following text \n{text}',
    input_variables=["text"]
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

report_gen_chain = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) >= 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'write an essay more than 500 words '}))