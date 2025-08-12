from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pathlib

def load_and_split_pdf(file_path: str | pathlib.Path):
    """
    Loads a PDF, extracts text, and splits it by paragraph.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of strings, where each string is a paragraph.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""]
        )
    
    docs = text_splitter.split_text(text)
    return docs

if __name__ == '__main__':
    # Example usage:
    pdf_path = pathlib.Path(__file__).parent / "test_docs" / "PaperQA.pdf"
    paragraphs = load_and_split_pdf(pdf_path)
    print(f"Successfully loaded and split {pdf_path.name}.")
    print(f"Number of paragraphs: {len(paragraphs)}")
    print("First paragraph:")
    print(paragraphs[0])