import logfire

def parse_text(file_path: str):
    with logfire.span("Text parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        except Exception as e:
            logfire.error(f"Error parsing text file: {str(e)}")
            return e
        