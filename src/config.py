import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root path
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "rag_eval.db"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Resolve Mistral API Key (handling MISTAL_API_KEY typo in the user's .env file)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTAL_API_KEY")

# Supported models
SUPPORTED_LLMS = [
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest"
]

SUPPORTED_EMBEDDINGS = [
    "mistral-embed",
    "bge-small-en-v1.5",
    "sentence-transformers-all-MiniLM-L6-v2"
]

# Pricing model (per token cost)
# Costs are listed in USD per token (derived from standard USD per 1M tokens)
# Mistral Large: $2 / 1M input, $6 / 1M output
# Mistral Medium: $2.7 / 1M input, $8.1 / 1M output
# Mistral Small: $1 / 1M input, $3 / 1M output
PRICING_CONFIG = {
    "llm": {
        "mistral-large-latest": {
            "input": 2.0 / 1_000_000,
            "output": 6.0 / 1_000_000
        },
        "mistral-medium-latest": {
            "input": 2.7 / 1_000_000,
            "output": 8.1 / 1_000_000
        },
        "mistral-small-latest": {
            "input": 1.0 / 1_000_000,
            "output": 3.0 / 1_000_000
        }
    },
    "embeddings": {
        "mistral-embed": 0.1 / 1_000_000,
        "bge-small-en-v1.5": 0.0,  # Local/free
        "sentence-transformers-all-MiniLM-L6-v2": 0.0  # Local/free
    }
}

def get_api_key() -> str:
    """Returns the validated Mistral API Key."""
    if not MISTRAL_API_KEY:
        raise ValueError("Mistral API Key not found. Please set MISTRAL_API_KEY or MISTAL_API_KEY in your .env file.")
    return MISTRAL_API_KEY
