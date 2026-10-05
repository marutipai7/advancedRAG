import logfire
from pypdf import PdfReader
import pdfplumber

def parse_pdf(file_path: str):
    with logfire.span("PDF Parsing", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages in {file_path}: {total_pages}")

            text_parts: list[str] = []
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)  # Store 1-based page number

            if blank_pages:
                logfire.info(f"pypdf returned empty text for pages: {blank_pages} in {file_path} - retrying with pdfplumber")

                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]  # Convert to 0-based index
                            fallback_text = page.extract_text() or ""
                            if fallback_text.strip():
                                text_parts.append(fallback_text)
                except Exception as plumber_error:
                    logfire.error(f"Error parsing PDF with pdfplumber: {str(plumber_error)}")

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                logfire.warning(f"Both pypdf and pdfplumber returned empty text for {file_path}, File may be fully image-based")

            else:
                logfire.info(f"Successfully parsed {file_path} with {len(full_text)} characters")

            return full_text
        
        except Exception as e:
            logfire.error(f"Error parsing PDF: {str(e)}")
            return e