import time  # Import time module to measure execution time
# Import necessary modules from langchain package
from langchain_community.document_loaders import (  # For loading documents from directories
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings  # For generating embeddings from documents
from langchain_community.vectorstores import FAISS  # For storing and searching vectors efficiently
from langchain_text_splitters import (  # For splitting text into manageable chunks
    RecursiveCharacterTextSplitter,
)

# Define paths for data and the FAISS database
DATA_PATH = "./data/"  # Path to the directory containing PDF documents
DB_FAISS_PATH = "vectorstore/db_faiss"  # Path to save the FAISS database

# Function to create a vector database using FAISS
def create_vector_db_faiss():
    start_time = time.time()  # Record the start time of the process
    print("Starting document loading process...")
    # Load documents from the specified directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

    # Load the documents using the loader
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    print("Splitting documents into smaller chunks...")

    # Split the loaded documents into smaller chunks for processing
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks")

    # Initialize embeddings model with a specified transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("Generating embeddings for document chunks...")

    # Create a FAISS database from the document chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)
    print("FAISS database created with document embeddings")

    # Save the FAISS database to the specified path
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS database saved to {DB_FAISS_PATH}")

# Main execution block
if __name__ == "__main__":
    start_time = time.time()  # Record the start time of the entire process
    print("Starting the vector database creation process...")
    create_vector_db_faiss()  # Call the function to create and save the FAISS database
    elapsed_time = time.time() - start_time  # Calculate total elapsed time
    print(f"Vector database creation process completed in {elapsed_time:.2f} seconds.")
