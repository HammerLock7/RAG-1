from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from ollama import chat
import streamlit as st

# Load documents
docs = PyPDFLoader("Thermochemistry5.1-5.2.pdf").load()
print(f"Number of documents loaded: {len(docs)}")

# Split documents
splitter = RecursiveCharacterTextSplitter (chunk_size=5000, chunk_overlap=1500)
chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
if chunks:
    print("First few chunk contents:")
    for i, chunk in enumerate(chunks[:3]): # Print the first 3 chunks
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:300]) # Print the first 300 characters
    print("-" * 20)

# Embeddings - TRY A DIFFERENT MODEL
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest") # Changed model
    test_embedding = embeddings.embed_query("This is a test sentence.")
    print(f"Test embedding length (with all-MiniLM-L6-v2): {len(test_embedding)}")
except Exception as e:
    print(f"Error during test embedding (with all-MiniLM-L6-v2): {e}")
    embeddings = None # Ensure embeddings is None if there's an error

if embeddings:
    # Vectorstore
    try:
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
        vectorstore.persist()

        def retrieve(query:str) -> str:
            results = vectorstore.similarity_search(query, k=3)
            return "\n\n".join([res.page_content for res in results]) # Corrected to use results

        def generate_answer(query:str, context:str) -> str:
            response = chat(
                model= "nomic-embed-text:latest", # Keeping the same generation model
                messages= [
                    {"role": "user", "content": f"Answer the question based on the context provided.\n\nContext {context}"},
                    {"role": "user", "content": query}
                ]
            )
            return response["message"]["content"]

        st.title("RAG demo")

        user_query = st.chat_input("Ask a question about Thermochemistry:")

        if user_query:
            with st.spinner("Retrieving relevant information..."):
                context = retrieve(user_query)

            with st.spinner("Generating answer..."):
                answer = generate_answer(user_query, context)

            st.write("**answer : **", answer)

    except ValueError as ve:
        st.error(f"Error creating vector store: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during vector store creation: {e}")
else:
    st.error("Embedding model failed to load. Please check the console for errors.")