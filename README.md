
# 🤖 GenAI & LangChain Playground — Multi-LLM Integration Project

Welcome to my LangChain-powered Generative AI projects!  
This repository demonstrates my hands-on implementation and learning of modern GenAI techniques using LangChain, integrating powerful LLMs such as **OpenAI GPT-4**, **Google Gemini**, **Groq (LLaMA, Mixtral, Gemma)**, and more.

---

## 📚 What I Learned and Implemented

- ✅ Learned to use **LangChain** to build modular GenAI applications from scratch.
- 🔗 Integrated multiple **LLM providers** — OpenAI, Gemini, Groq, Claude — using LangChain wrappers and chains.
- 🔍 Implemented **Retrieval-Augmented Generation (RAG)** pipelines using FAISS and Chroma for knowledge-grounded responses.
- ✨ Explored and applied **prompt engineering**, system messages, templates, and memory-based chains.
- 🧠 Developed smart assistants with **ConversationalRetrievalChain**, **MultiQueryRetriever**, and **LangChain Agents**.
- 🧾 Loaded and chunked documents (PDF, TXT, HTML) to allow natural language querying over user-provided data.

---

## 🚀 Use Cases Demonstrated

- 🧑‍💼 Chatbot with context-aware responses across multiple models.
- 📄 Question answering from uploaded PDFs or local documents.
- 🧠 Research assistant with semantic search and summarization.
- 🤖 LLM Agent that uses tools and reasoning for complex tasks.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **LangChain** | Framework for LLM chaining and orchestration |
| **OpenAI GPT-4** | Natural language understanding & generation |
| **Google Gemini** | Advanced text + multimodal capabilities |
| **Groq** | High-speed local model inference (LLaMA, Mixtral, etc.) |
| **FAISS / Chroma** | Vector stores for RAG |
| **Python** | Backend logic and orchestration |
| **Streamlit / CLI** | Simple interface for testing chains |

---

## 📂 Project Structure

```bash
.
├── app.py                 # Streamlit or CLI demo
├── chains/               # LangChain chains and agents
├── prompts/              # Prompt templates
├── loaders/              # Custom document loaders
├── utils/                # Helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project overview (this file)
```

---

## ⚙️ Quickstart Guide


1. **Set up the environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scriptsctivate
pip install -r requirements.txt
```

2. **Add your API keys**
Create a `.env` file:
```env
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
GROQ_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

3. **Run the app**
```bash
python app.py
# or streamlit run app.py
```

---

## 🧠 Example Chains You’ll Find Inside

- `ConversationalRetrievalChain` with memory
- `MultiQueryRetriever` for diverse document retrieval
- `LLMChain` with system prompts
- Tool-using `AgentExecutor` with function-calling LLMs

---


## 📌 Tags & Keywords

`#LangChain` `#LLM` `#GenAI` `#OpenAI` `#GoogleGemini` `#Groq` `#PromptEngineering` `#RAG` `#AIChatbot` `#Python`

---

## 🤝 Acknowledgments

Thanks to the LangChain community and documentation for making LLM integration easy and extensible.
