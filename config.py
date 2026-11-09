"""הגדרות מרכזיות של המערכת. כל הערכים ניתנים לדריסה במשתני סביבה / קובץ .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv אופציונלי
    pass

# שימוש במאגר התעודות של Windows (נחוץ ברשתות עם סינון כמו נטפרי, שמחליפות תעודות HTTPS).
# חייב לרוץ לפני כל חיבור לרשת, ולכן הוא כאן.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# --- נתיבים ---
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "family_events")

# --- מודלים ---
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini").lower()
_DEFAULT_EMBED = {"gemini": "gemini-embedding-001", "local": "BAAI/bge-m3"}
EMBED_MODEL = os.getenv("EMBED_MODEL", _DEFAULT_EMBED.get(EMBED_PROVIDER, ""))
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "")
EMBED_TEXT_PREFIX = os.getenv("EMBED_TEXT_PREFIX", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-5-5"}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- אחזור ---
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0"))
# --- חלוקה ל-Chunks ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- שדות בתחילת כל פריט ---
FIELD_LABELS = {"item_id": "מזהה", "item_type": "סוג", "event_name": "שם השמחה"}

# --- תצוגה ---
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "500"))
NO_INFO_MESSAGE = "אין לי מידע על כך במסמכי המשפחה."