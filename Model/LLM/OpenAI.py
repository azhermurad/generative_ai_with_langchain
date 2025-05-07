from langchain_openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    api_key=openai_api_key
    )

result = llm.invoke("what is nlp?")


print(result)