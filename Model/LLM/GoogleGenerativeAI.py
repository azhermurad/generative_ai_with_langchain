from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)


result = llm.invoke("AI stands for?")

print(result)




