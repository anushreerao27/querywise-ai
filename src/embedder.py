from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store(chunks):
   
    print("Loading embedding model... (first run downloads ~90MB, be patient!)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating FAISS vector store from chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the vector store to disk so we don't re-embed every time
    vector_store.save_local("faiss_index")
    print("Vector store saved to faiss_index/")

    return vector_store


def load_vector_store():
    
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("No FAISS index found. Please upload and process a PDF first.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded from disk.")
    return vector_store