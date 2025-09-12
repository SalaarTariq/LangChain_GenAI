from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import  RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)


parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)


# Create a lambda that preserves the original input and adds sentiment
def preserve_input_and_add_sentiment(input_dict):
    sentiment_result = classifier_chain.invoke(input_dict)
    return {
        'sentiment': sentiment_result.sentiment,
        'feedback': input_dict['feedback']
    }

# Create the branch chain that works with the combined data
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] == 'positive', prompt2 | model | parser),
    (lambda x: x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# Chain that combines sentiment analysis with response generation
chain = RunnableLambda(preserve_input_and_add_sentiment) | branch_chain

print(chain.invoke({'feedback': 'This is a terrible phone'}))