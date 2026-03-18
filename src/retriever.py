from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


def load_llm():
    print("Loading LLM via Groq...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )
    print("LLM loaded!")
    return llm


def build_qa_chain(vector_store):
    llm = load_llm()

    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever


def ask_question(qa_chain, retriever, question: str):
    print(f"\nQuestion: {question}")
    answer = qa_chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources