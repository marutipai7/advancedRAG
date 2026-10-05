from bs4 import BeautifulSoup, Tag, soup
import logfire

def parse_html(file_path: str):
    with logfire.span("HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'html.parser')
            for script in soup[Tag](['script', 'style', 'noscript', 'meta']):
                script.decompose()  # Remove script and style elements

            text = soup.get_text(separator='\n')

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            chunks = (phrase.strip() for line in lines for phrase in line.split("  ") if phrase.strip())
            text_cleaned = '\n'.join(chunk for chunk in chunks if chunk)

            return text_cleaned
        except Exception as e:
            logfire.error(f"Error parsing HTML: {str(e)}")
            return e