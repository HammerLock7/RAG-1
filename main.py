from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from ollama import chat
from pdf2image import convert_from_path
import pytesseract
import os
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama


# config
PDF_PATH = "Thermochemistry5.1-5.2.pdf"
POPLER_PATH = r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\tesseract.exe"
DB_DIR = "chroma_db"
TXT_DUMP = "ocr_text.txt"

#FastAPI setup
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#Pydantic Model
class QueryRequest(BaseModel):
    question: str

#Embedding Model

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Load or Build VectorStore

def extract_text_from_pdf():
    """Extracts OCR text from PDF and saves to disk to avoid repeating this step."""
    print("Extracting text from PDF via OCR...")
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    images = convert_from_path(PDF_PATH, poppler_path=POPLER_PATH)
    all_text = "\n".join([pytesseract.image_to_string(img) for img in images])
    with open(TXT_DUMP, "w", encoding="utf-8") as f:
        f.write(all_text)
    return all_text

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("Loading existing vectorstore...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    print("Building vectorstore for the first time...")

    if os.path.exists(TXT_DUMP):
        with open(TXT_DUMP, "r", encoding="utf-8") as f:
            all_text = f.read()
    else:
        all_text = extract_text_from_pdf()

    docs = [Document(page_content=all_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # Less overlap = faster
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    vectorstore.persist()
    print("Vectorstore built and saved.")

retriever = vectorstore.as_retriever()

#Retrieval + Generation Logic

def retrieve(query, retriever, k=2):
    """Returns top-k relevant context chunks for the user's question."""
    docs = retriever.invoke(query)[:k]
    context = "\n\n".join(doc.page_content for doc in docs)
    print("\n--- Retrieved Context ---")
    print(context)
    print("-------------\n")
    return context
    

def generate_answer(query, context, model_name="gemma:2b"):
    """ Generate an answer from the retrieved context and query using ollama."""
    prompt_template = PromptTemplate.from_template(
        "Answer the question based only on the context below:\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    ollama_model = ChatOllama(model=model_name, temperature=0.3)
    chain = prompt_template | ollama_model | StrOutputParser()

    answer= chain.invoke({
        "context": context,
        "question": query
    })

    print("\n--- Generated Answer---")
    print(answer)
    print("----------------\n")

    return answer

from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate_answer(query: str, context: str) -> str:
    return generate_answer(query, context)


#Web Routes

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def query_api(payload: QueryRequest):
    query = payload.question
    context = retrieve(query, retriever)
    answer = generate_answer(query, context)
    return {"answer": answer}
