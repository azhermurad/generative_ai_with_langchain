import os
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes


from dotenv import load_dotenv
load_dotenv()



# loading env varables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")



# create the chatllm 

llm = ChatGroq( model="gemma2-9b-it")

prompt = ChatPromptTemplate.from_messages([
    ("system","Translate the following sentence to {language}: Return only translated text"),
    ("user","{text}")
])


parser = StrOutputParser()


chain = prompt |  llm | parser

# print(chain.invoke({"language":"urdu","text":"what's your name?"}))


# setup sever logic

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain"
)

@app.get("/")
def read_root():
    return {"Hi": "Aly Api Is working!!!!"}

add_routes(
    app, 
    chain, 
    path="/translate"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    

