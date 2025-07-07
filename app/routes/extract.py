from fastapi import APIRouter, UploadFile, File
import fitz, uuid, os, ollama
from app.chromadb_store import store_embedding
from app.config import Config

router = APIRouter()

@router.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_filename = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_filename, "wb") as f:
            f.write(contents)

        doc = fitz.open(temp_filename)
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        os.remove(temp_filename)

        response = ollama.embeddings(
            model=Config.EMBEDDING_MODEL,
            prompt=full_text
        )
        embedding = response["embedding"]
        doc_id = str(uuid.uuid4())

        store_embedding(doc_id, full_text, embedding, {
            "filename": file.filename,
            "document_id": doc_id
        })

        return {
            "message": "PDF extracted successfully",
            "filename": file.filename,
            "document_id": doc_id,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}
