from pathlib import Path
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,load_index_from_storage,StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

import chromadb

BASE_DIR = "./"
PERSIST_DIR = BASE_DIR + "/data" + "/storage" 
DATA_DIR = BASE_DIR + "/data" +"/raw" + "/french"



# Initialize the LLM with optimized settings
embed_model = OllamaEmbedding(
    model_name="embeddinggemma",
    request_timeout=300.0,  # Increased timeout for large documents
)
llm = Ollama(
    model="llama3.2:latest",  # Confirm with `ollama list`
    request_timeout=300.0,
    temperature=0.1,          # Lower temperature for more factual responses
)

# Set global configurations
Settings.embed_model = embed_model
Settings.llm = llm

def load_and_index_documents(persist_dir=PERSIST_DIR, data_dir=DATA_DIR, embed_model=embed_model):
    """Load documents and create vector index"""

    persist_file = persist_dir + "/docstore.json"
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if os.path.exists(persist_file):
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir),
        )
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage...")
    else:

        # Check if data directory exists
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add your files.")

        # Load documents from the data folder
        
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        docs = SimpleDirectoryReader(data_dir).load_data()
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                TitleExtractor(),
                embed_model(),
            ],
            vector_store=vector_store,
        )
        pipeline.run(documents=docs, 
                    num_workers=4)

        if not docs:
            raise ValueError(f"No documents found in {data_dir}")


        # Build vector index from documents
        index = VectorStoreIndex.from_vector_store(vector_store)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print("Index created and persisted to storage...")

    return index

def create_query_engine(index, similarity_top_k=3):
    """Create query engine with specified retrieval parameters"""

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,  # Number of relevant chunks to retrieve
        response_mode="compact"             # Compact response generation
    )

    return query_engine


def create_chat_engine(index,similarity_top_k=3):
    """Create a chat engine with memory"""
    memory = ChatSummaryMemoryBuffer.from_defaults(token_limit=3000)
    chat_engine = index.as_chat_engine(
        llm=llm,
        memory=memory,
        similarity_top_k=similarity_top_k,
        chat_mode="condense_plus_context",  # Allows the model to use previous messages as context
        context_prompt=(
       "You are a chatbot that answers questions about French history, but you can also have normal conversations."
        "Here are the French history documents you need to use to answer the questions about French history:\n"
        "{context_str}\n"
        "Instructions: Answer the questions about French history accurately using the documents above."
        "When you are unable to answer a question about French history, say you don't know."
        "Answer in the same language as the last question asked."
    ),
        verbose=True
    )
    return chat_engine

def get_chat_engine(similarity_top_k=3):
    index = load_and_index_documents()
    return create_chat_engine(index, similarity_top_k)
