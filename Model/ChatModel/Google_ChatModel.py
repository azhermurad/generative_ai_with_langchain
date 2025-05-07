from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os



load_dotenv()
print(os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
)

result = llm.invoke("ai stands for?")
print(result)