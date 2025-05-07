import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)




st.title("Large Language Models")
text_input = st.text_input(
        "Enter some text 👇"
    )

result = llm.invoke(text_input)  

if text_input:
    st.write("You entered: ", text_input)

st.markdown(
  f""" 
   {result}
    """
)
