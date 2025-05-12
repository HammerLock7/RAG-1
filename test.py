from langchain_community.document_loaders import TextLoader

file_path = "C:/Users/lebak/Documents/RAG-1/data.txt"  # change this to match your file
loader = TextLoader(file_path)

docs = loader.load()
print(docs[0].page_content)

def retrieve(query:str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        vectorstore = chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")