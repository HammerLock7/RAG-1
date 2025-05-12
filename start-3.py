import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.chains import retrieval_qa
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


#Load documents
loader = TextLoader("data1.txt")
docs = loader.load()

# create a vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Set-up a retriever and QA chain
retriever = db.as_retriever()
llm = OpenAI(temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# streamlit UI
st.title("RAG CHATBOT")

query= st.text_input("Ask a question")
if query:
    answer= qa_chain.run(query)
    st.write("Answer:",answer)