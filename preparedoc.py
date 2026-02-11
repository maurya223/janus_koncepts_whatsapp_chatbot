from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load your document
loader = TextLoader("my_document.txt")  # your file
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_KEY")

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore")

print("Documents indexed successfully!")
