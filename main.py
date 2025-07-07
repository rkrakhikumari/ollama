from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import fitz  
import ollama
import uuid
import os
from chromadb_store import store_embedding, query_similar_documents, list_documents

app = FastAPI()

class QueryInput(BaseModel):
    query: str
    document_id: str

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2" 

@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        temp_filename = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_filename, "wb") as f:
            f.write(contents)

        doc = fitz.open(temp_filename)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        os.remove(temp_filename)
        
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=full_text
        )

        embedding = response["embedding"]
        doc_id = str(uuid.uuid4())

        store_embedding(doc_id, full_text, embedding, metadata={
            "filename": file.filename,
            "document_id": doc_id
        })

        return {
            "message": "pdf extrcted sucessfuly",
            "filename": file.filename,
            "document_id": doc_id,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/query/")
async def query_for_answers(input: QueryInput):
    try:
        query_embedding_response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=input.query
        )
        
        query_embedding = query_embedding_response["embedding"]

        results = query_similar_documents(
            input.query,
            query_embedding,
            filter_doc_id=input.document_id
        )

        matched_docs = results.get("documents", [[]])
        
        if matched_docs and matched_docs[0]:
            context = matched_docs[0][0]
            
            prompt = f"""answer the question based on context below.

Context: {context}

Question: {input.query}

Answer:"""

            answer_response = ollama.generate(
                model=LLM_MODEL,
                prompt=prompt,
                stream=False
            )
            
            return {
                "query": input.query,
                "answer": answer_response["response"],
                "document_id": input.document_id,
                "status": "success"
            }
        
        return {
            "query": input.query,
            "answer": "No relevant information found in the document.",
            "document_id": input.document_id,
            "status": "no_match"
        }
    
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/docs-list/")
def get_all_documents():
    try:
        docs = list_documents()
        return {
            "documents": docs,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}