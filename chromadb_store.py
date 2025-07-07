import chromadb
from typing import Optional, Dict, Any

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_embeddings")

def store_embedding(document_id: str, text: str, embedding: list[float], metadata: Optional[Dict[str, Any]] = None):
    try:
        if metadata is None:
            metadata = {}
        
        metadata["document_id"] = document_id
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[document_id],
            metadatas=[metadata]
        )
        return True
    except Exception as e:
        return False

def list_documents():
    try:
        result = collection.get(include=["documents", "metadatas"])
        
        formatted_docs = []
        if result.get("ids"):
            for i, doc_id in enumerate(result["ids"]):
                doc_info = {
                    "document_id": doc_id,
                    "metadata": result["metadatas"][i] if result.get("metadatas") else {},
                    "text_preview": result["documents"][i][:200] + "..." if len(result["documents"][i]) > 200 else result["documents"][i]
                }
                formatted_docs.append(doc_info)
        
        return {
            "total_documents": len(formatted_docs),
            "documents": formatted_docs
        }
    except Exception as e:
        return {"error": str(e), "total_documents": 0, "documents": []}

def query_similar_documents(query_text: str, embedding: list[float], filter_doc_id: str = None, n_results: int = 3):
    try:
        query_args = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter_doc_id:
            query_args["where"] = {"document_id": filter_doc_id}

        results = collection.query(**query_args)
        return results

    except Exception as e:
        return {"error": str(e), "documents": [[]], "metadatas": [[]], "distances": [[]]}