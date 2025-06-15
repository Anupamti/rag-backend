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

from typing import List
from langchain_core.documents import Document

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    template = """Answer the question based only on the following context:
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
