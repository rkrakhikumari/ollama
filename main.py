from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  
import requests
import uuid
import re  

from chromadb_store import store_embedding, query_similar_documents, list_documents

app = FastAPI()

class QueryInput(BaseModel):
    query: str
    document_id: str

def extract_relevant_sentences(text: str, query: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'[.!?●]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    query_lower = query.lower()
    relevant_sentences = []
    
    for sentence in sentences:
        if query_lower in sentence.lower():
            relevant_sentences.append(sentence.strip())
    
    if not relevant_sentences:
        for sentence in sentences:
            words = sentence.lower().split()
            if any(query_lower in word for word in words):
                relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        return ". ".join(relevant_sentences[:max_sentences])
    else:
        return "No relevant sentences found."

@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    doc = fitz.open("temp.pdf")
    full_text = "".join([page.get_text() for page in doc])
    doc.close()

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": full_text}
    )

    if response.ok:
        embedding = response.json()["embedding"]
        doc_id = str(uuid.uuid4())

        store_embedding(doc_id, full_text, embedding)

        return {
            "filename": file.filename,
            "text": full_text,
            "embedding_length": len(embedding),
            "document_id": doc_id
        }

    return {"error": response.text}

@app.get("/docs-list/")
def get_all_documents():
    docs = list_documents()
    return JSONResponse(content=docs)

@app.post("/query/")
def semantic_search(input: QueryInput):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": input.query}
    )
    if not response.ok:
        return {"error": "Failed to generate embedding from query"}

    query_embedding = response.json()["embedding"]

    results = query_similar_documents(
        input.query,
        query_embedding,
        filter_doc_id=input.document_id
    )

    matched_docs = results.get("documents", [[]])
    
    if matched_docs and matched_docs[0]:
        full_text = matched_docs[0][0]
        relevant_text = extract_relevant_sentences(full_text, input.query)
        
        return {
            "query": input.query,
            "matched_text": relevant_text
        }
    
    return {
        "query": input.query,
        "matched_text": "No match found."
    }