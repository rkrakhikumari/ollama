import chromadb
from typing import Optional, Dict, Any

client = None
collection = None

def init_chroma(db_path: str, collection_name: str):
    global client, collection
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

def store_embedding(document_id: str, text: str, embedding: list[float], metadata: Optional[Dict[str, Any]] = None):
    if collection is None:
        raise RuntimeError("Chroma collection not initialized. Call init_chroma first.")
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

def list_documents():
    if collection is None:
        raise RuntimeError("Chroma collection not initialized. Call init_chroma first.")
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

def query_similar_documents(query_text: str, embedding: list[float], filter_doc_id: str = None, n_results: int = 3):
    if collection is None:
        raise RuntimeError("Chroma collection not initialized. Call init_chroma first.")
    query_args = {
        "query_embeddings": [embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }
    if filter_doc_id:
        query_args["where"] = {"document_id": filter_doc_id}
    results = collection.query(**query_args)
    return results
