from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os



load_dotenv()

llm = ChatOpenAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key = os.getenv("OPENAI_API_KEY")
    
)

result = llm.invoke("ai stands for?")
print(result)