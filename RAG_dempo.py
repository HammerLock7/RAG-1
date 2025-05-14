from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat
import streamlit as st

docs = PyPDFLoader("Thermochemistry5.1-5.2.pdf").load()
print(f"number of documents loaded:{len(docs)}")

splitter = RecursiveCharacterTextSplitter (chunk_size=5000, chunk_overlap=1500)
chunks = splitter.split_documents(docs)
print(f"number of chunks created: {len(chunks)}")
if chunks:
    print("First chunk content:")
    print(chunks[0].page_content[:200]) #print first 200 characters of the chunks content

embeddings = OllamaEmbeddings(model="tazarov/all-minilm-l6-v2-f32:latest")

try:
    test_embedding = embeddings.embed_query("This is a test sentence.")
    print(f"Test embedding length: {len(test_embedding)}")
except Exception as e:
    print(f"Error during test embedding: {e}")

vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")


def retrieve(query:str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([docs.page_content for docs in results])

def generate_answer(query:str, context:str) -> str:
    response = chat(
        model= "tazarov/all-minilm-l6-v2-f32:latest",
        messages= [
            {"role": "user", "content": f"Answer the question based on the context provided.\n\nContext {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response["message"]["content"]

st.title("RAG demo")

user__query = st.chat_input("Ask a question about Thermochemistry:")

if user__query:
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(user__query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(user__query, context)

    st.write("**answer : **", answer)