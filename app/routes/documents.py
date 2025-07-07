from fastapi import APIRouter
from app.chromadb_store import list_documents

router = APIRouter()

@router.get("/docs-list/")
def get_all_documents():
    try:
        docs = list_documents()
        return {"documents": docs, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
