from bs4 import BeautifulSoup
import logfire


def parse_html(file_path: str):
    with logfire.span("HTML Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")

            # Remove non-content elements
            for element in soup.find_all(
                ["script", "style", "noscript", "meta"]
            ):
                element.decompose()

            # Extract visible text
            text = soup.get_text(separator="\n")

            # Clean empty lines and whitespace
            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            # Further split content where multiple spaces are used
            chunks = (
                phrase.strip()
                for line in lines
                for phrase in line.split("  ")
                if phrase.strip()
            )

            text_cleaned = "\n".join(
                chunk for chunk in chunks if chunk
            )

            return text_cleaned

        except Exception as e:
            logfire.error(
                f"Error parsing HTML: {type(e).__name__}: {str(e)}"
            )
            return ""