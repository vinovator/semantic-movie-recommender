# src/config.py

import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# --- File Paths ---
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DB_DIR = "db"
PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "movies_master_with_plots.parquet")

# --- Model & Search Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 10 # Default number of results to return

# --- LLM Configuration ---

# Google Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL= "gemini-1.5-flash-latest"

# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# Set the provider: 'gemini' or 'ollama'
LLM_PROVIDER = "ollama"
# LLM_PROVIDER = "gemini"
