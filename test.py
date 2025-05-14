from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat
import streamlit as st
from pdf2image import convert_from_path
from langchain.schema import Document
import pytesseract

# Convert PDF pages to images
images = convert_from_path(
    "Thermochemistry5.1-5.2.pdf",
    poppler_path=r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"
)


# Extract text with OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\tesseract.exe"
all_text = ""
for i, img in enumerate(images):
    text = pytesseract.image_to_string(img)
    print(f"Page {i+1} text length: {len(text)}")
    all_text += text + "\n"

# Create a LangChain-compatible Document object
docs = [Document(page_content=all_text)]
print("Total extracted text length:", len(all_text))

#split to chunks
splitter = RecursiveCharacterTextSplitter (chunk_size=5000, chunk_overlap=1500)
chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
if chunks:
    print("First few chunk contents:")
    for i, chunk in enumerate(chunks[:3]): # Print the first 3 chunks
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:300]) # Print the first 300 characters
    print("-" * 20)


# Embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    test_embedding = embeddings.embed_query("This is a test sentence.")
    print(f"Test embedding length: {len(test_embedding)}")
except Exception as e:
    print(f"Embedding model load error: {e}")
    embeddings = None

if not chunks:
    raise ValueError("No document chunks created. Check PDF content and splitter settings.")

try:
    test = embeddings.embed_query("Test sentence")
    print("Test embedding created. Vector length:", len(test))
except Exception as e:
    raise ValueError(f"Embedding model failed: {e}")

# Embed each chunk manually for debug
texts = [chunk.page_content for chunk in chunks]
embeddings_list = embeddings.embed_documents(texts)
print(f"Created {len(embeddings_list)} embeddings.")

if not embeddings_list:
    raise ValueError("No embeddings returned. Check model and input text.")

if embeddings:
    try:
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
        vectorstore.persist()

        def retrieve(query: str) -> str:
            results = vectorstore.similarity_search(query, k=3)
            return "\n\n".join([res.page_content for res in results])

        def generate_answer(query: str, context: str) -> str:
            response = chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}"},
                    {"role": "user", "content": query}
                ]
            )
            return response["message"]["content"]

        st.title("RAG DEMO")

        user_query = st.chat_input("Ask a question about Thermochemistry:")
        if user_query:
            with st.spinner("Retrieving relevant information..."):
                context = retrieve(user_query)

            with st.spinner("Generating answer..."):
                answer = generate_answer(user_query, context)

            st.write("**Answer:**", answer)

    except Exception as e:
        st.error(f"Unexpected error during vectorstore or retrieval: {e}")
else:
    st.error("Embedding model failed to load. Please check the console for more info.")
