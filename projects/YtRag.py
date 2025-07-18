# import os
# from fastapi import FastAPI
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import YoutubeLoader
# from langchain_community.document_loaders.youtube import TranscriptFormat
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# from dotenv import load_dotenv

# load_dotenv()


# # loading env varables
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# # create the chatllm

# llm = ChatGroq(model="gemma2-9b-it")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# from langchain.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an intelligent assistant that answers user questions "
#             "based on the transcript of a YouTube video. Always refer only to the transcript content. "
#             "If the answer is not present in the transcript, respond with 'The information is not available in the video.'",
#         ),
#         (
#             "human",
#             "Here is the video transcript:\n\n{context}\n\n"
#             "Now, based on the video, answer the following question:\n{question}",
#         ),
#     ]
# )


# parser = StrOutputParser()




# # loader
# loader = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=QsYGlZkevEg",
#     transcript_format=TranscriptFormat.CHUNKS,
#     chunk_size_seconds=300,
# )

# trans_docs = loader.load()

# # vector store

# vector_store = Chroma(
#     collection_name="embedding_collection",
#     embedding_function=embeddings,
#     # persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
# )
# vector_store.add_documents(documents=trans_docs)
# retiver = vector_store.as_retriever(search_kwargs={"k":2})


# from langchain_core.runnables import RunnableParallel,RunnablePassthrough

# chain = RunnableParallel({
#     "context": retiver,
#     "question": RunnablePassthrough()
# }) | prompt | llm | parser


# print(chain.invoke("what is the video about?"))





import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM & Embedding

llm = ChatGroq(model="gemma2-9b-it")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an intelligent assistant that answers user questions based on the transcript of a YouTube video. "
     "Always refer only to the transcript content. If the answer is not present, respond with "
     "'The information is not available in the video.'"),
    ("human", 
     "Here is the video transcript:\n\n{context}\n\n"
     "Now, based on the video, answer the following question:\n{question}")
])
parser = StrOutputParser()

# Streamlit page config
st.set_page_config(page_title="🎥 YouTube Video Q&A", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## Settings")
    st.info("This tool uses **RAG**, Google AI Embeddings, and Groq's Gemma model.")
    st.markdown("---")
    st.markdown("**Developer:** [Azher Ali](https://github.com/azhermurad)")
    st.markdown("**Contact:** azheraly009@example.com")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>YouTube Video Q&A Assistant 🎥🤖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Ask anything from a YouTube video using the power of RAG (Retrieval-Augmented Generation)</p>", unsafe_allow_html=True)
st.markdown("---")

# Inputs
col1, col2 = st.columns([3, 2])
with col1:
    video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    question = st.text_input("❓ Your Question", placeholder="What is the main topic of the video?")

# Main logic
if video_url and question:
    with st.spinner("🔍 Processing video and preparing response..."):
        try:
            # Load transcript
            loader = YoutubeLoader.from_youtube_url(
                video_url,
                transcript_format=TranscriptFormat.CHUNKS,
                chunk_size_seconds=300,
                language=["en", "en-US"]
            )
            trans_docs = loader.load()

            # Embedding + Vector store
            vector_store = Chroma(
                embedding_function=embeddings,
            )
            vector_store.add_documents(trans_docs)
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})

            # RAG chain
            chain = (
                RunnableParallel({
                    "context": retriever,
                    "question": RunnablePassthrough()
                }) | prompt | llm | parser
            )

            # Output
            answer = chain.invoke(question)

            st.success("✅ Answer ready!")
            st.markdown("### 💬 Answer")
            st.markdown(f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; font-size: 18px;'>{answer}</div>", unsafe_allow_html=True)

            with st.expander("📜 View Raw Transcript Chunks"):
                for i, doc in enumerate(trans_docs[:5]):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content[:500]}...")
        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.info("👆 Enter a YouTube video link and a question to get started.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Azher Ali. All rights reserved.</p>", unsafe_allow_html=True)
