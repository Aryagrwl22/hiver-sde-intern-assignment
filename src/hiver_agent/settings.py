import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "mock_key_if_missing")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "mock_key_if_missing")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_INDEX_SIZE = int(os.environ.get("RETRIEVAL_INDEX_SIZE", 250)) # Lowered to 250 to avoid free tier rate limits, configurable via env
