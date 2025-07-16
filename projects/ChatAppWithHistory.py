# How to add message history
# https://python.langchain.com/v0.2/docs/how_to/message_history/

import os

# from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# from langserve import add_routes
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()


# loading env varables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# create the chatllm

llm = ChatGroq(model="gemma2-9b-it")

prompt = ChatPromptTemplate(
    [
        SystemMessage(
            content="You are a helpful assistant. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder("history"),
        ("human", "{messages}"),
        # pass a list of message
    ]
)

# history of the llm
message = [
    HumanMessage(content="my name is azher ali"),
    AIMessage(
        content="Hello Azher Ali, Nice to meet you! 😊 Is there anything I can help you with today?"
    ),
]

chain = prompt | llm | StrOutputParser()
# response = chain.invoke({"history": message, "messages": "how do you know my name?"})
# print(response)


# How to store and load messages


# A key part of this is storing and loading messages. When constructing RunnableWithMessageHistory you need to pass in a get_session_history function. This function should take in a session_id and return a BaseChatMessageHistory object.

# What is session_id?

# session_id is an identifier for the session (conversation) thread that these input messages correspond to. This allows you to maintain several conversations/threads with the same chain at the same time.

# What is BaseChatMessageHistory?

# BaseChatMessageHistory is a class that can load and save message objects. It will be called by RunnableWithMessageHistory to do exactly that. These classes are usually initialized with a session id.

# Let's create a get_session_history object to use for this example. To keep things simple, we will use a simple SQLiteMessage


from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
    ChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# now we have to define the
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


my_self_into = """ 
Hi there, I'm Azher Ali!
Data Scientist | AI/Ml Engineer | Data Analyst | Backend Developer | Open-Source Enthusiast
I'm a passionate Data Scientist and AI/ML Engineer with hands-on experience building intelligent systems using machine learning, deep learning, and natural language processing. I specialize in developing scalable models using frameworks like PyTorch, TensorFlow, and Python — from research to real-world deployment."""

# runnerwtihHistroy

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the questions",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

response = runnable_with_history.invoke(
    {"input": my_self_into},
    config={"configurable": {"session_id": "1"}},
)


response = runnable_with_history.invoke(
    {"input": "who am i?"},
    config={"configurable": {"session_id": "1"}},
)

query = runnable_with_history.invoke(
    {"input": "what is the name of person we are taking?"},
    config={"configurable": {"session_id": "1"}},
)

print(query)
