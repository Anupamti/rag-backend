import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import shutil
from typing import List, Dict, Any
from langchain_core.documents import Document
import json

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

llm = ChatOpenAI(model=MODEL_NAME)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Separate collections for different document types
DOCUMENT_COLLECTION = "documents"
TRANSCRIPT_COLLECTION = "transcripts"
persist_directory = "./chroma_db"

# Initialize vectorstores
document_vectorstore = None
transcript_vectorstore = None

# Initialize vectorstores if they exist
if os.path.exists(persist_directory):
    try:
        document_vectorstore = Chroma(
            collection_name=DOCUMENT_COLLECTION,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    except:
        document_vectorstore = None
    
    try:
        transcript_vectorstore = Chroma(
            collection_name=TRANSCRIPT_COLLECTION,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    except:
        transcript_vectorstore = None

# Different text splitters for different content types
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""]
)

transcript_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_document(file_path: str) -> List[Document]:
    """Load document based on file extension"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise Exception(f"Unsupported file format: {file_path}")
    return loader.load()

def process_uploaded_document(file_path: str):
    """Process policy documents, service descriptions, etc."""
    global document_vectorstore
    
    docs = load_document(file_path)
    
    # Add metadata to identify as policy/service document
    for doc in docs:
        doc.metadata.update({
            "doc_type": "document",
            "source_file": os.path.basename(file_path),
            "collection": DOCUMENT_COLLECTION
        })
    
    splits = document_splitter.split_documents(docs)
    
    if document_vectorstore:
        document_vectorstore.add_documents(splits)
    else:
        document_vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=DOCUMENT_COLLECTION,
            persist_directory=persist_directory
        )

def process_uploaded_transcript(file_path: str, metadata: Dict[str, Any] = None):
    """Process client transcripts with metadata"""
    global transcript_vectorstore
    
    docs = load_document(file_path)
    
    # Add transcript-specific metadata
    for doc in docs:
        doc.metadata.update({
            "doc_type": "transcript",
            "source_file": os.path.basename(file_path),
            "collection": TRANSCRIPT_COLLECTION
        })
        
        # Add session metadata if provided
        if metadata:
            doc.metadata.update({
                "client_id": metadata.get("client_id"),
                "session_date": metadata.get("session_date"),
                "case_manager": metadata.get("case_manager"),
                "filename": metadata.get("filename")
            })
    
    splits = transcript_splitter.split_documents(docs)
    
    if transcript_vectorstore:
        transcript_vectorstore.add_documents(splits)
    else:
        transcript_vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=TRANSCRIPT_COLLECTION,
            persist_directory=persist_directory
        )

def docs2str(docs):
    """Convert documents to string with metadata context"""
    result = []
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata
        
        if metadata.get("doc_type") == "transcript":
            prefix = f"[TRANSCRIPT - Client: {metadata.get('client_id', 'Unknown')}, Date: {metadata.get('session_date', 'Unknown')}, Case Manager: {metadata.get('case_manager', 'Unknown')}]"
        else:
            prefix = f"[DOCUMENT - Source: {metadata.get('source_file', 'Unknown')}]"
        
        result.append(f"{prefix}\n{content}")
    
    return "\n\n".join(result)

def get_relevant_context(question: str, query_type: str = "general"):
    """Get relevant context based on query type"""
    all_docs = []
    
    # Determine which collections to search based on query type
    if query_type == "policy_check":
        # For policy compliance questions, prioritize documents but include relevant transcripts
        if document_vectorstore:
            doc_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 4})
            all_docs.extend(doc_retriever.get_relevant_documents(question))
        
        if transcript_vectorstore:
            transcript_retriever = transcript_vectorstore.as_retriever(search_kwargs={"k": 2})
            all_docs.extend(transcript_retriever.get_relevant_documents(question))
    
    elif query_type == "recommendation":
        # For recommendations, prioritize transcripts and service documents
        if transcript_vectorstore:
            transcript_retriever = transcript_vectorstore.as_retriever(search_kwargs={"k": 3})
            all_docs.extend(transcript_retriever.get_relevant_documents(question))
        
        if document_vectorstore:
            doc_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 3})
            all_docs.extend(doc_retriever.get_relevant_documents(question))
    
    elif query_type == "cross_reference":
        # For cross-referencing across transcripts
        if transcript_vectorstore:
            transcript_retriever = transcript_vectorstore.as_retriever(search_kwargs={"k": 6})
            all_docs.extend(transcript_retriever.get_relevant_documents(question))
    
    else:  # general queries
        # Search both collections equally
        if document_vectorstore:
            doc_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 3})
            all_docs.extend(doc_retriever.get_relevant_documents(question))
        
        if transcript_vectorstore:
            transcript_retriever = transcript_vectorstore.as_retriever(search_kwargs={"k": 3})
            all_docs.extend(transcript_retriever.get_relevant_documents(question))
    
    return all_docs

def get_rag_response(question: str, query_type: str = "general") -> str:
    """Enhanced RAG response with query type awareness"""
    global document_vectorstore, transcript_vectorstore
    
    if not document_vectorstore and not transcript_vectorstore:
        return "No documents or transcripts have been uploaded yet. Please upload some content first."
    
    # Get relevant context based on query type
    relevant_docs = get_relevant_context(question, query_type)
    
    if not relevant_docs:
        return "No relevant context found for your question."
    
    # Choose prompt template based on query type
    if query_type == "policy_check":
        template = """You are an assistant that checks if case managers followed proper policies and procedures.

Use the following context which includes policy documents and transcript excerpts:

{context}

Question: {question}

Please analyze whether the case manager's actions in the transcript align with the policies provided. Be specific about:
1. Which policies are relevant
2. Whether the case manager followed or violated them
3. Any recommendations for improvement

Answer:"""

    elif query_type == "recommendation":
        template = """You are an assistant that provides service recommendations based on client needs and available services.

Use the following context which includes client transcripts and available services:

{context}

Question: {question}

Based on what the client discussed in their session(s) and the services available, provide specific recommendations:
1. What services would benefit this client
2. Why these services are appropriate
3. Any evidence-based practices that apply

Answer:"""

    elif query_type == "cross_reference":
        template = """You are an assistant that analyzes information across multiple client sessions.

Use the following context from multiple transcript sessions:

{context}

Question: {question}

Analyze the information across the different sessions, noting:
1. Common themes or issues that appear in multiple sessions
2. Changes or progress over time
3. Any concerning patterns or positive developments

Answer:"""

    else:  # general
        template = """You are an assistant who answers questions using the following context.
Context may include policy documents, service descriptions, evidence-based practices, state standards, and transcripts of conversations with clients.

Always cite the relevant source when possible and distinguish between information from documents vs. client transcripts.

{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": lambda x: docs2str(relevant_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)

def clear_docs_folder(folder_path="docs"):
    """Clear all uploaded files"""
    if not os.path.exists(folder_path):
        return "Docs folder does not exist."
    
    deleted = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                deleted.append(filename)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    return f"Deleted files: {deleted}" if deleted else "No files to delete."

def reset_context():
    """Reset all vector stores and clear files"""
    global document_vectorstore, transcript_vectorstore
    
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    document_vectorstore = None
    transcript_vectorstore = None
    clear_docs_folder()

def get_collection_stats():
    """Get statistics about the collections"""
    stats = {
        "documents": 0,
        "transcripts": 0,
        "total_chunks": 0
    }
    
    if document_vectorstore:
        try:
            doc_collection = document_vectorstore._collection
            stats["documents"] = doc_collection.count()
        except:
            stats["documents"] = "Unknown"
    
    if transcript_vectorstore:
        try:
            transcript_collection = transcript_vectorstore._collection
            stats["transcripts"] = transcript_collection.count()
        except:
            stats["transcripts"] = "Unknown"
    
    if isinstance(stats["documents"], int) and isinstance(stats["transcripts"], int):
        stats["total_chunks"] = stats["documents"] + stats["transcripts"]
    
    return stats