from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

#load environment variables from the .env file
load_dotenv()

#Load documents
docs = PyPDFLoader("Thermochemistry5.1-5.2.pdf").load()

#split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1500)
chunks = splitter.split_documents(docs)

# create a vector store
embeddings = OllamaEmbeddings(model="llama3.2")
db = FAISS.from_documents(chunks, embeddings)

# Set-up a retriever and QA chain
retriever = db.as_retriever()
llm = Ollama(model="llama3.2",temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# streamlit UI
st.title("RAG CHATBOT")

query= st.text_input("Ask a question")
if query:
    answer= qa_chain.run(query)
    st.write("Answer:",answer)