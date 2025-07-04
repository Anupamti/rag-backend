import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import shutil


from typing import List
from langchain_core.documents import Document

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

llm = ChatOpenAI(model=MODEL_NAME)

embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
collection_name = "my_collection"
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
else:
    vectorstore = None  # initialized after first upload

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_document(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise Exception("Unsupported file format")
    return loader.load()

def process_uploaded_file(file_path: str):
    global vectorstore

    docs = load_document(file_path)
    splits = text_splitter.split_documents(docs)
    
    if vectorstore:
        vectorstore.add_documents(splits)
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_response(question: str) -> str:
    global vectorstore

    if vectorstore is None:
        return "No documents have been uploaded yet. Please upload a document first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    template = """You are an assistant who answers questions using the following context. 
    Context may include policy documents, transcripts of conversations with clients, or case manager notes. 
    Always cite relevant information when possible.
    {context}
    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | docs2str, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

def clear_docs_folder(folder_path="docs"):
    if not os.path.exists(folder_path):
        return "Docs folder does not exist."

    deleted = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                deleted.append(filename)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return f"Deleted files: {deleted}" if deleted else "No files to delete."



def reset_context():
    global vectorstore
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    vectorstore = None
    clear_docs_folder()

