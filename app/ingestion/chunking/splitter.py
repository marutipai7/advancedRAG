import logfire
from typing import List


def chunk_text(text: str, chunk_size:int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 1500.

    Returns:
        List[str]: A list of text chunks.
    """
    with logfire.span("Chunking text", chunk_size=chunk_size, text_length=len(text)):
        if not text.split():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        logfire.info(f"Chunking complete. Generated {len(valid_chunks)} chunks.")
        return valid_chunks