from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile
from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embedder import create_vector_store, load_vector_store
from src.retriever import build_qa_chain, ask_question

st.set_page_config(
    page_title="QueryWise AI",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 QueryWise AI")
st.caption("Upload a PDF and ask questions about it — powered by RAG + LangChain + Groq")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading and indexing your PDF..."):
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = load_pdf(temp_path)
                chunks = chunk_documents(pages)
                vector_store = create_vector_store(chunks)
                qa_chain, retriever = build_qa_chain(vector_store)

                st.session_state.qa_chain = qa_chain
                st.session_state.retriever = retriever
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name

            st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")
            st.info("Ask questions below!")

    if st.session_state.pdf_name:
        st.divider()
        st.caption(f"Active doc: {st.session_state.pdf_name}")
        if st.button("Clear & upload new PDF"):
            st.session_state.qa_chain = None
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.rerun()

    st.divider()
    st.caption("Built by Anushree Rao")
    st.caption("Stack: LangChain · FAISS · Groq · Streamlit")

if st.session_state.qa_chain is None:
    st.info("Upload a PDF in the sidebar to get started!")
    st.markdown("""
    **How it works:**
    1. Upload any PDF document
    2. The app chunks and indexes it using FAISS vector store
    3. Ask questions in natural language
    4. Get answers grounded in your document with source references
    """)
else:
    # Display chat history with sources
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View source chunks used"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1}** — Page {doc.metadata.get('page', 0) + 1}")
                    st.caption(doc.page_content)
                    st.divider()

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Use only the current question for retrieval
        # but pass history to the LLM for context awareness
        history_context = ""
        recent = [m for m in st.session_state.chat_history[-6:] 
                  if m["role"] in ["user", "assistant"]]
        for msg in recent[:-1]:  # exclude current question
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"

        # Retrieve using clean question only
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant chunks using clean question
                sources = st.session_state.retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in sources)

                # Build full prompt with history for the LLM
                from langchain_core.messages import HumanMessage, SystemMessage
                from langchain_groq import ChatGroq
                import os

                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    api_key=os.getenv("GROQ_API_KEY")
                )

                prompt = f"""You are a helpful assistant. Answer using only the document context below.
If the answer is not in the context, say "I couldn't find that in the document."

Document context:
{context}

Previous conversation:
{history_context}

Current question: {question}

Answer:"""

                response = llm.invoke([HumanMessage(content=prompt)])
                answer = response.content

            st.write(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })