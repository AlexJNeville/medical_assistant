import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. SETUP PATHS
# Using the path you provided earlier
file_path = r"C:\Users\alexn\Coding\Lucas\Seperate funny little parts\your_illness_data\data.txt"
persist_directory = "./medical_db"

print("🚀 Starting Medical AI Indexing...")

# 2. LOAD DATA
try:
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"✅ File loaded successfully.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# 3. CHUNK DATA
# This turns your one big text file into searchable paragraphs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Increased from 600
    chunk_overlap=500,      # Keeps context between chunks
    separators=["\n\n", "\n", ". ", " "] # Prioritizes splitting at paragraphs
)
chunks = text_splitter.split_documents(documents)
print(f"📝 Split data into {len(chunks)} searchable chunks.")

# 4. EMBEDDINGS
# This will download a small model (~80MB) on the first run
print("🧠 Initializing embeddings (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. CREATE DATABASE
# This creates a folder on your computer so you don't have to re-process every time
print("💾 Building the local database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=persist_directory
)

print("✨ ALL SET! Your data is now indexed.")

# 6. TEST QUERY
query = "What is the primary illness mentioned in the data?"
results = vector_db.similarity_search(query, k=1)

if results:
    print("\n--- TEST SEARCH RESULT ---")
    print(results[0].page_content)
else:
    print("\n⚠️ No results found. Check if your data.txt has content.")
