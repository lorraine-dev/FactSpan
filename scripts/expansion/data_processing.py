from langdetect import detect, LangDetectException
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_author_data(raw_json):
    """Extracts author data from the raw JSON dictionary."""
    try:
        author_data = raw_json.get('author', {})
        return (
            author_data.get('@type'),
            author_data.get('name'),
            author_data.get('url')
        )
    except Exception as e:
        logging.error(f"Error processing author data: {e}")
        return None, None, None

def detect_language(text):
    """Detects the language of the given text."""
    try:
        if isinstance(text, str) and len(text.strip()) > 0:
            return detect(text)
        else:
            return "unknown"
    except LangDetectException:
        return "unknown"
