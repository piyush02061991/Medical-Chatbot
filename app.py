from flask import Flask, render_template, request
from SRC.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Validate keys
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("❌ Missing API keys in .env file")

# Initialize embeddings
embeddings = download_embeddings()

# Connect to Pinecone index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM
chatModel = ChatOpenAI(model="gpt-4")

# Build retrieval chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatModel,
    retriever=retriever
)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")  # Make sure templates/chat.html exists

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        print("User message:", msg)

        # Provide both required keys
        response = rag_chain.run({
            "question": msg,
            "chat_history": []  # Empty for now; can be extended later
        })

        print("Response:", response)
        return str(response)
    except Exception as e:
        print("❌ Error in chat route:", e)
        return "Error: " + str(e)

# Run server
if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
