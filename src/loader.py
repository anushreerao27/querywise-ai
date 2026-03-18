import os
import pdfplumber

def load_pdf(file_path: str):
    """
    Uses pdfplumber instead of PyPDFLoader —
    much better at handling real-world PDFs,
    preserves proper spacing and layout.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at: {file_path}")

    from langchain_core.documents import Document

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text and text.strip():
                pages.append(Document(
                    page_content=text.strip(),
                    metadata={"source": file_path, "page": i}
                ))

    print(f"Loaded '{file_path}' — {len(pages)} pages found.")
    return pages