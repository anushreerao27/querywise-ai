# 🔍 QueryWise AI

A RAG-based PDF chatbot — upload any document and ask questions in natural language.
Built with LangChain, FAISS, Groq (Llama3), and Streamlit.

## Demo
![QueryWise AI Demo](demo.gif)

## How it works
1. Upload any PDF document
2. Text is extracted, chunked, and embedded into a FAISS vector store
3. Ask questions in natural language
4. Relevant chunks are retrieved and passed to Llama3 via Groq API
5. Grounded answers are returned with source chunk attribution

## Architecture
```
PDF → pdfplumber → text chunks → FAISS embeddings
User question → similarity search → top-k chunks → Groq LLM → answer
```

## Tech Stack
| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Llama3 via Groq API |
| UI | Streamlit |
| PDF Parsing | pdfplumber |

## Run locally
```bash
git clone https://github.com/anushreerao27/querywise-ai.git
cd querywise-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run:
```bash
streamlit run app.py
```

## Features
- Clean PDF text extraction using pdfplumber
- Semantic search using FAISS vector store
- Conversation memory — follow-up questions work correctly
- Source chunk attribution for every answer
- Works with any text-based PDF

## Built by
Anushree Rao — CSE (AI & ML), RNS Institute of Technology