from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Set environment variables explicitly (if needed)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

# Define index name
index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)



# Create a retriever for similarity search
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize OpenAI model
llm = OpenAI(temperature=0.4, max_tokens=500)

# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg  # Renamed 'input' to avoid overwriting built-in Python function
    print(input_text)

    response = rag_chain.invoke({"input": msg})  # Fixed incorrect quotes
    print("Response:", response["answer"])

    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


