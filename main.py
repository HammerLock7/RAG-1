from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ollama import chat
from pdf2image import convert_from_path
import pytesseract
import os

#FastAPI setup
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#Model
class Query(BaseModel):
    question : str

#OCR & Langchain setup 
print ("Extracting PDF...")
images = convert_from_path(
    "Thermochemistry5.1-5.2.pdf",
    poppler_path=r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"
)

pytesseract.pytesseract.tesseract_cmd =  r"C:\Program Files\tesseract.exe"
all_text= "\n".join([pytesseract.image_to_string(img)for img in images])
docs = [Document(page_content=all_text)]

print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=750)
chunks = splitter.split_documents(docs)

print("Embedding chunks...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
vectorstore.persist()

def retrieve(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([res.page_content for res in results])

def generate_answer(query: str, context: str) -> str:
    response = chat(
        model = "llama3.2",
        messages=[
            {"role": "system", "content": "Answer the question based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}"},
            {"role": ":user", "content": query}
        ]
    )
    return response["message"]["content"]

#Web routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(data: Query):
    context = retrieve(data.question)
    answer=generate_answer(data.question, context)
    return {"answer": answer}