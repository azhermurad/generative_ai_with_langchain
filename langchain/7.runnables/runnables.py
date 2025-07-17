
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnableParallel

import streamlit as st

# loading env varables
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.title("Runnables in Langchain")
# create the chatllm 
llm = ChatGroq( model="gemma2-9b-it")

prompt = ChatPromptTemplate.from_messages([
    ("system","Your are a helper assistance, you have to help users in ask questions"),
    ("user","generate a linkedin post on {topic} topic")
])


prompt2 = ChatPromptTemplate.from_messages([
    ("user","make a professional two line from this text {text} and Please make hashtag for this")
])




parser = StrOutputParser()

 # <-------------------------------------------------- RunnableSequence ------------------------------------------------------------------>

# def add_one(x):
#     st.write(x)
    
#     st.write("enddddddddddddddddddddddddddddddddddddddddd")
    
#     return x 

# runnable_1 = RunnableLambda(add_one)

# chain = RunnableSequence(prompt,llm, parser,prompt2,runnable_1, llm, parser)

# input_str = st.text_input("What is in your mind")

# if input_str:
#     st.write(chain.invoke({"topic":input_str}))







 # <-------------------------------------------------- RunnableParallel ------------------------------------------------------------------>
 
 
joke_prompt =  ChatPromptTemplate.from_messages([
     ("human","tell me a joke about {topic}")
])

 
poem_prompt =  ChatPromptTemplate.from_messages([
     ("human","write a 2-line poem about {topic}")
])



# chain = RunnableParallel({
#     "joke": RunnableSequence(joke_prompt | llm),
#     "poem": RunnableSequence(poem_prompt | llm)
# })


# st.write(chain.invoke({"topic":"programming"}))







# <-------------------------------------------------- RunnableBranch ------------------------------------------------------------------>



# conducation runnable in langchain


prompt_one = ChatPromptTemplate.from_messages(
    [
        ("human","write a professional description on this topic: \n\n Topic: {topic}")
        
    ]
)

# st.write(prompt_one.invoke({"topic":"langchain"}))


from langchain_core.runnables import RunnableBranch




branch = RunnableBranch(
    (lambda x: len(x) >500, lambda x: x.upper()),
     lambda x: x
    
)


chain = prompt_one | llm | parser | branch

response = chain.invoke({"topic":"ai"})

st.write(response)