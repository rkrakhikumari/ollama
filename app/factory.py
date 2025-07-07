from fastapi import FastAPI
from app.routes import extract, query, documents
from app.config import Config
from app.chromadb_store import init_chroma

def create_app() -> FastAPI:
    app = FastAPI(title="PDF QA API")

    # Initialize DB connection
    init_chroma(Config.CHROMA_DB_PATH, Config.COLLECTION_NAME)

    # Include routes
    app.include_router(extract.router)
    app.include_router(query.router)
    app.include_router(documents.router)

    return app
