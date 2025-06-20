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

def determine_source_type(filename: str, content: str) -> str:
    """Determine if the document is a transcript or regular document based on filename and content."""
    filename_lower = filename.lower()
    content_lower = content.lower()
    
    # Check filename for transcript indicators
    transcript_keywords_filename = ['transcript', 'conversation', 'interview', 'call', 'meeting']
    if any(keyword in filename_lower for keyword in transcript_keywords_filename):
        return "transcript"
    
    # Check content for formal transcript indicators
    transcript_keywords_content = ['speaker:', 'client:', 'interviewer:', 'interviewee:', 'caller:', 'agent:']
    if any(keyword in content_lower for keyword in transcript_keywords_content):
        return "transcript"
    
    # Check for conversational patterns typical of transcripts
    conversational_indicators = [
        'yes, sir', 'no, sir', 'okay', 'yeah', 'right', 'good', 'all right',
        'i\'m going to start the recording', 'any questions', 'how\'s', 'still'
    ]
    
    # Count conversational indicators
    indicator_count = sum(1 for indicator in conversational_indicators if indicator in content_lower)
    
    # Check for question-answer patterns (look for multiple question marks)
    question_count = content.count('?')
    
    # Check for short responses typical of dialogue
    lines = content.split('\n')
    short_lines = [line.strip() for line in lines if len(line.strip()) < 20 and len(line.strip()) > 0]
    
    # If we have multiple conversational indicators, questions, and short responses, likely a transcript
    if indicator_count >= 3 and question_count >= 5 and len(short_lines) >= 10:
        return "transcript"
    
    return "document"

def load_document(file_path: str) -> List[Document]:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise Exception("Unsupported file format")
    
    # Load documents
    docs = loader.load()
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    
    # Determine source type based on content of first document (representative of the whole file)
    sample_content = docs[0].page_content if docs else ""
    source_type = determine_source_type(filename, sample_content)
    
    # Add metadata to all documents
    for doc in docs:
        if not hasattr(doc, 'metadata') or doc.metadata is None:
            doc.metadata = {}
        doc.metadata['source'] = source_type
        doc.metadata['filename'] = filename
    
    return docs

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