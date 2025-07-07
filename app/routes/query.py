from fastapi import APIRouter
from pydantic import BaseModel
import ollama
from app.chromadb_store import query_similar_documents
from app.config import Config

router = APIRouter()

class QueryInput(BaseModel):
    query: str
    document_id: str

@router.post("/query/")
async def query_for_answers(input: QueryInput):
    try:
        response = ollama.embeddings(
            model=Config.EMBEDDING_MODEL,
            prompt=input.query
        )
        embedding = response["embedding"]

        results = query_similar_documents(input.query, embedding, input.document_id)
        matched_docs = results.get("documents", [[]])

        if matched_docs and matched_docs[0]:
            context = matched_docs[0][0]
            prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {input.query}\n\nAnswer:"

            answer = ollama.generate(model=Config.LLM_MODEL, prompt=prompt, stream=False)["response"]

            return {
                "query": input.query,
                "answer": answer,
                "document_id": input.document_id,
                "status": "success"
            }

        return {
            "query": input.query,
            "answer": "No relevant information found.",
            "document_id": input.document_id,
            "status": "no_match"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}
