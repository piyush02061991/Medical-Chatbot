# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
# OLD
# from langchain.schema import Document

# NEW
from langchain_core.documents import Document



#Extract text from pdf files
def load_pdf_files(data):
    loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document object, return a new list of documents object
    containing only the 'source' metadata field and original page_content.
    """
    minimal_docs: List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    
    return minimal_docs



#split documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

#using Higging face embedding model to create embeddings
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings=download_embeddings()