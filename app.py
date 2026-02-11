from flask import Flask, request
import requests
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# OpenAI API setup
os.environ["OPENAI_API_KEY"] = os.getenv("GROK_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.x.ai/v1"

# Load or create FAISS vector store
try:
    if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
    else:
        loader = PyPDFLoader("documents/Janus Koncepts – Ai Chatbot Knowledge Base (google Docs Ready).pdf")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")
        print("Documents indexed successfully!")

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=db.as_retriever())
except Exception as e:
    print(f"Error loading vector store: {e}")
    qa = None

# Flask app
app = Flask(__name__)

GROK_API_KEY = os.getenv("GROK_API_KEY")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

@app.route('/')
def home():
    return "WhatsApp Chatbot is running!"

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "verification failed", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    print("Received data:", data)

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]
        text = message["text"]["body"]

        response = qa.run(text) if qa else "I'm not ready yet. Try again later."
        send_message(sender, response)
    except Exception as e:
        print("Error:", e)
        send_message(sender, "Sorry, I couldn't process your message.")
    return "ok", 200

def send_message(recipient, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "content-type": "application/json"
    }
    payload = {"messaging_product": "whatsapp", "to": recipient, "text": {"body": text}}
    requests.post(url, json=payload, headers=headers)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
