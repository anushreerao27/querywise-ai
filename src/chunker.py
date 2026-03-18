from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(pages):
   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    print(f"Split into {len(chunks)} chunks.")
    return chunks