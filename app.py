from flask import Flask, request
import requests
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables locally (Render reads from dashboard)
load_dotenv()

# Flask app
app = Flask(__name__)

# Environment variables
GROK_API_KEY = os.getenv("GROK_API_KEY")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Set OpenAI API key (used by LangChain)
os.environ["OPENAI_API_KEY"] = GROK_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.x.ai/v1"

# Load or create FAISS vector store
try:
    embeddings = OpenAIEmbeddings()
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
    else:
        loader = PyPDFLoader("documents/Janus Koncepts – Ai Chatbot Knowledge Base (google Docs Ready).pdf")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")
        print("Documents indexed successfully!")

    # Set up QA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
except Exception as e:
    print(f"Error loading vector store: {e}")
    qa = None

# Home route
@app.route('/')
def home():
    return "WhatsApp Chatbot is running!"

# Webhook verification
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    else:
        return "verification failed", 403

# Webhook to receive messages
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print("Received data:", data)

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]
        text = message["text"]["body"]

        # Get AI response
        if qa is not None:
            response = qa.run(text)
        else:
            response = "I'm sorry, I cannot answer right now. Please try again later."

        send_message(sender, response)

    except KeyError as e:
        print("KeyError:", e)
    except Exception as e:
        print("Error:", e)
        send_message(sender, "Sorry, I couldn't process your message.")

    return "ok", 200

# Function to send WhatsApp message
def send_message(recipient, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "text": {"body": text}
    }
    requests.post(url, json=payload, headers=headers)

# Main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
