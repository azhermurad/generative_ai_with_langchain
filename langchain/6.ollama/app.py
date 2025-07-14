import os
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# loading env varables
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")




prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. please response to the question asked.",
        ),
        ("human", "Question: {question}"),
    ]
)

# streamlit ui
st.set_page_config(layout="wide")
st.title("ChatOllama Model")
input_text = st.text_input("Enter your question here:")



llm = ChatOllama(model="llama3.2:1b")
parser = StrOutputParser()

chain = prompt | llm | parser

if input_text:
    st.write(chain.invoke({"question": input_text}),b)
    
    
    
    
    





