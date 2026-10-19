from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None):
    """Splits incoming documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)