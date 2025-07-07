import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_embeddings")
    DEBUG = os.getenv("DEBUG", "True") == "True"
