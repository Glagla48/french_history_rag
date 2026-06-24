from src.config import (PERSIST_DIR, RAW_DATA_DIR, 
                        get_ollama_embed_model, 
                        get_ollama_llm, 
                        SYSTEM_PROMPT_PATH, 
                        CONTEXT_PROMPT_PATH)
from src.rag.indexing import load_and_index_documents
from src.rag.postprocessor import (get_SentenceTransformerRerank, 
                                   get_SentenceEmbeddingOptimizer, 
                                   get_SimilarityPostprocessor)
from src.rag.retriever import get_QueryFusionRetriever

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Settings


Settings.llm = get_ollama_llm()
Settings.embed_model = get_ollama_embed_model()

def get_chat(persist_dir=PERSIST_DIR, 
             data_dir=RAW_DATA_DIR, 
             embed_model=Settings.embed_model, 
             llm=Settings.llm,
             similarity_top_k:int=5) :

    index = load_and_index_documents(embed_model, persist_dir, data_dir)
    memory = ChatSummaryMemoryBuffer.from_defaults(token_limit=3000)
    
    reranker = get_SentenceTransformerRerank()
    response_synthesizer = get_response_synthesizer(llm=llm, 
                                                    verbose=True, 
                                                    use_async=True)
    retriever = get_QueryFusionRetriever(index, 5, similarity_top_k)

    with open(CONTEXT_PROMPT_PATH, "r" ) as f:
        context_prompt = f.read()

    with open(SYSTEM_PROMPT_PATH, "r" ) as f:
        system_prompt = f.read()
    
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        node_postprocessors=[
            reranker,
            get_SentenceEmbeddingOptimizer(),
            get_SimilarityPostprocessor()
        ],
        response_synthesizer=response_synthesizer,
        verbose=True,
        context_prompt=context_prompt,
        system_prompt=system_prompt
    )
    return chat_engine, retriever
