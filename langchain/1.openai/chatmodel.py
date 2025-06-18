import getpass
import os
from langchain_openai import ChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


print(os.environ.get("OPENAI_API_KEY"))


# Chatopenai method automatically find teh api key with the name of OPENAI_API_KEY
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


ai_msg = llm.invoke("what is deep learning?")
ai_msg