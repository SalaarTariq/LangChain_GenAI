from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)


model2 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes rom the following text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate short and simple mcqs from following text \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n -> {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()
parallel = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel | merge_chain
 
text = """Machine Learning (ML) is one of the most exciting fields of technology today. It powers many of the applications we use every day, from Netflix recommending movies, to Google Maps predicting traffic, to banks detecting fraud in real-time. At its core, Machine Learning is about teaching computers to learn from data and improve over time without being explicitly programmed. Unlike traditional rule-based programming where instructions are fixed, ML systems learn patterns from data and use them to make predictions or decisions. For example, instead of coding a rule like “If salary > 100,000 then approve loan,” we can feed the computer thousands of past loan applications with outcomes, and the ML model will learn hidden patterns to predict whether a new application should be approved or not.

The importance of Machine Learning lies in its ability to automate tasks, personalize experiences, predict future outcomes, and scale to handle massive amounts of data. It reduces human effort in repetitive processes, tailors recommendations like YouTube and Spotify playlists, forecasts events such as weather or stock movements, and supports critical domains like healthcare, finance, and security. This is why ML is often described as the engine that drives modern Artificial Intelligence.

Machine Learning can be broadly categorized into three main types. Supervised learning involves training models with labeled data, where both inputs and correct outputs are provided, allowing the model to predict outcomes for new inputs. For instance, predicting house prices or detecting spam emails are common supervised tasks. Unsupervised learning, on the other hand, deals with unlabeled data and focuses on finding hidden structures or patterns, such as segmenting customers into groups based on their behavior. Finally, reinforcement learning is inspired by how humans and animals learn, where an agent improves its actions through trial and error, guided by rewards and penalties. Famous applications include self-driving cars and AI systems like AlphaGo that defeated world champions in the game of Go.

The process of Machine Learning usually follows a series of steps. First, data is collected from various sources such as images, text, or numerical records. This data is then preprocessed — cleaned, formatted, and prepared to remove errors and inconsistencies. Afterward, an appropriate model or algorithm is selected, such as decision trees, neural networks, or regression models. The model is then trained by feeding it the prepared data, and afterward, it is evaluated on unseen data to measure accuracy and reliability. Once the model performs well, it is deployed into real-world applications where it can continuously provide predictions and insights.

Some of the most widely used algorithms in Machine Learning include linear regression for predicting continuous values like house prices, logistic regression for classification problems such as yes/no decisions, and decision trees or random forests that can handle both regression and classification tasks effectively. More advanced algorithms like support vector machines, clustering methods, and deep neural networks expand ML’s reach into areas like image recognition, speech processing, and natural language understanding.

In summary, Machine Learning is transforming the way technology interacts with our lives. From everyday conveniences like recommendation systems to critical applications in medicine and self-driving cars, ML is shaping the present and defining the future. Its power lies in learning from data, improving with experience, and making intelligent predictions — and as data continues to grow, the role of Machine Learning will only become more significant."""

print(chain.invoke({'text':text}))