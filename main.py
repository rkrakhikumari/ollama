from fastapi import FastAPI, File, UploadFile
import fitz  

app = FastAPI()

@app.post("/extract-text/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    doc = fitz.open("temp.pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    return {"filename": file.filename, "text": full_text}




