from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve and set Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load and process PDF data
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medicalbot"

# Create Pinecone index
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Embed each chunk and upsert the embeddings into Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

