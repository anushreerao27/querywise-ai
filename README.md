# 🔍 QueryWise AI

> A production-deployed RAG (Retrieval-Augmented Generation) chatbot — upload any PDF and ask questions in natural language, powered by LangChain, FAISS, Groq (Llama3), and Streamlit.

🚀 **[Live Demo](https://querywise-ai-yysvqsnfqmyguuc9iwrghm.streamlit.app)**

---

## What is this?

QueryWise AI lets you upload any PDF document and have a conversation with it. Instead of reading through pages manually, just ask questions and get grounded answers — with the exact source chunks shown for every response.

Built using a **RAG (Retrieval-Augmented Generation)** pipeline:
- Your PDF is chunked and converted into vector embeddings
- When you ask a question, the most relevant chunks are retrieved via semantic search
- Those chunks + your question are passed to Llama3 (via Groq) to generate a grounded answer
- The source chunks used are displayed for full explainability

---

## Demo

![QueryWise AI Demo](demo.gif)

*(Upload a PDF → Ask questions → Get grounded answers with source attribution)*

---

## Features

- **Clean PDF parsing** using pdfplumber — handles real-world documents correctly
- **Semantic search** using FAISS vector store + sentence-transformers embeddings
- **Conversation memory** — follow-up questions maintain context
- **Source attribution** — every answer shows which chunks were used
- **Works with any text-based PDF** — research papers, resumes, reports, textbooks
- **Production deployed** on Streamlit Cloud

---

## Architecture

```
PDF Upload
    │
    ▼
pdfplumber (text extraction)
    │
    ▼
RecursiveCharacterTextSplitter (chunking)
    │
    ▼
all-MiniLM-L6-v2 (embeddings)
    │
    ▼
FAISS Vector Store (indexing)
    │
    ▼
User Question → Similarity Search → Top-3 Chunks
    │
    ▼
Groq API / Llama3-70b (answer generation)
    │
    ▼
Answer + Source Chunks (Streamlit UI)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Llama3-70b via Groq API |
| UI | Streamlit |
| PDF Parsing | pdfplumber |
| Deployment | Streamlit Cloud |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/anushreerao27/querywise-ai.git
cd querywise-ai
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
querywise-ai/
│
├── src/
│   ├── loader.py        # PDF loading with pdfplumber
│   ├── chunker.py       # Text splitting
│   ├── embedder.py      # Embeddings + FAISS vector store
│   └── retriever.py     # QA chain with Groq LLM
│
├── app.py               # Streamlit UI
├── requirements.txt     # Dependencies
└── .env                 # API keys (not committed)
```

---

## Key Concepts Demonstrated

- **RAG pipeline** — end-to-end implementation from PDF ingestion to answer generation
- **Vector similarity search** — FAISS indexing and retrieval
- **LLM prompt engineering** — context-grounded answering with fallback handling
- **Conversation memory** — history-aware question answering
- **MLOps** — environment management, secrets handling, cloud deployment
- **Clean code architecture** — separation of concerns across modules

---

## Built By

**Anushree Rao**
Third-year CSE (AI & ML) student at RNS Institute of Technology, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-anushreerao27-black?logo=github)](https://github.com/anushreerao27)
