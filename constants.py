import os
from chromadb.config import Settings

MODELS_PATH = "./models"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=False,
)

# Max New Tokens
MAX_NEW_TOKENS = 4096

EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)


MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_BASENAME = None