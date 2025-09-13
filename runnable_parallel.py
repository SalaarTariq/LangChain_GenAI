from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel , RunnableSequence
from dotenv  import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

prompt1= PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)


prompt2= PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

paralle_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1 , model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})


result = paralle_chain.invoke({'topic':'AI'})
print(result)