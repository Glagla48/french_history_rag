import os
import sys
from pathlib import Path
from shutil import rmtree

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, load_index_from_storage,StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(PROJECT_ROOT)


from src.rag.pipeline import chunk_documents_with_metadata, get_ingestion_pipeline
from src.config import RAW_DATA_DIR, PERSIST_DIR, get_ollama_embed_model, get_ollama_llm


def load_and_index_documents(embed_model,
                            persist_dir:Path=PERSIST_DIR, 
                             data_dir:Path=RAW_DATA_DIR, 
                             worker:int=1,
                             pipeline:bool=False
                             ):

    """Load documents and create vector index"""
    if os.path.exists(persist_dir / "docstore.json"):
        print("Load Index From Storage")
        db = chromadb.PersistentClient(path=str(persist_dir))
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir),
            vector_store=vector_store,
        )
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("Index loaded from storage...")
        print("docstore len : ", len(index.docstore.docs))
    else:
        if persist_dir.exists():
            rmtree(persist_dir)
            print(f"🧹 Dossier {persist_dir} supprimé.")
        else:
            print("Dossier créé : ", persist_dir)
            os.makedirs(persist_dir, exist_ok=True)

        print("Creating Index From documents")
        # Check if data directory exists
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add your files.")

        docs = SimpleDirectoryReader(data_dir).load_data()
        if not docs:
            raise ValueError(f"No documents found in {data_dir}")

        print("Extracting document metadata and chunking...")
        splitter = SentenceSplitter(chunk_overlap=150, chunk_size=1024)
        if pipeline:
            pipe = get_ingestion_pipeline()
            nodes = splitter(docs)
            """nodes = pipe.run(nodes=nodes, 
                             cache_collection=True,
                             show_progress=True,
                             num_workers=worker)"""
        else:
            nodes = chunk_documents_with_metadata(docs, splitter, show_progress=True)
        
        print(f"Pipeline finished — {len(nodes)} chunks from {len(docs)} documents")

        # Build vector index from documents
        print("Builing Storage Context and VectorStore")
        db = chromadb.PersistentClient(path=str(persist_dir))
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        index.storage_context.persist(persist_dir=str(persist_dir))
        print("docstorez len : ", len(index.docstore.docs))
        print("Index created and persisted to storage...")

    return index

def main():
    Settings.embed_model = get_ollama_embed_model()
    Settings.llm = get_ollama_llm()

    load_and_index_documents(Settings.embed_model, pipeline=True)

if __name__ == "__main__":
    main()
