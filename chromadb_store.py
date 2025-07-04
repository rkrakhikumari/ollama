import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pprint import pprint

# Create a persistent Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection
collection = client.get_or_create_collection(name="pdf_embeddings")

# Updated function signature to match your current usage
def store_embedding(document_id: str, text: str, embedding: list[float]):
    try:
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[document_id],
            metadatas=[{"document_id": document_id}]  # Important for filtering
        )
        print(f"[INFO] Stored document {document_id}")
    except Exception as e:
        print(f"[ERROR] Failed to store document {document_id}: {e}")

def list_documents():
    try:
        result = collection.get(include=["documents", "embeddings", "metadatas"])
        pprint(result)
        return result
    except Exception as e:
        print(f"[ERROR] Could not retrieve documents: {e}")
        return {"error": str(e)}

def query_similar_documents(query_text: str, embedding: list[float], filter_doc_id: str = None):
    try:
        query_args = {
            "query_embeddings": [embedding],
            "n_results": 3,
            "include": ["documents", "metadatas", "distances", "embeddings"]
        }

        if filter_doc_id:
            query_args["where"] = {"document_id": filter_doc_id}  # ChromaDB filter

        results = collection.query(**query_args)
        print(f"[INFO] Similarity search results: {results}")
        return results

    except Exception as e:
        print(f"[ERROR] Failed similarity query: {e}")
        return {"error": str(e)}