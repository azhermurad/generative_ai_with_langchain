import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter


# loading env varables
from dotenv import load_dotenv

load_dotenv()


# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# llm model
llm = ChatGroq(model="llama-3.1-8b-instant")

# # embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", show_progress=True
)


# PDF loading
loader = PyPDFDirectoryLoader("./papers")
docs = loader.load()


# # Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# vectorstore 
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# # prompt
# print(len(splits))

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer 
            the question. If you don't know the answer, say that you 
            don't know.
            
            {context}
            
            Question: {input}

            Helpful Answer:
            """,
        ),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(rag_chain.invoke({"input": "What is the BART?"})["answer"])


# RAG
# documents load --> split --> embeddings --> store in vector-db --> retriever
