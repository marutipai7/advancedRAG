import logfire
from unstructured.partition.auto import partition

def parse_office(file_path: str):
    with logfire.span("Office Parsing", fiolename=file_path):
        try:
            elements = partition(file_path)
            full_text = "\n".join(str(element) for element in elements)

            if not full_text.strip():
                logfire.warning("Unstructured returned empty text for {file_path}")

            else:
                logfire.info(f"Successfully parsed {file_path} with Unstructured with {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"Error parsing Office file: {str(e)}")
            return e