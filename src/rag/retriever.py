from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import Stemmer

QUERY_GEN_PROMPT = (
    "Tu es un assistant spécialisé en histoire de France. "
    "Génère {num_queries} variantes de recherche en FRANÇAIS pour la question suivante.\n"
    "Réponds UNIQUEMENT avec les variantes, une par ligne, sans numérotation.\n"
    "Question : {query}\n"
)

def get_bm25_retriver(index, topk:int):
    
    return BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=topk, # Add filters here
        stemmer=Stemmer.Stemmer("french"),
        language="french")

def get_QueryFusionRetriever(index, num_queries:int, topk:int)->QueryFusionRetriever:
    return QueryFusionRetriever([
         index.as_retriever(similarity_top_k=topk),
        get_bm25_retriver(index, topk)
    ], 
    num_queries=num_queries, 
    verbose=True,
    use_async=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
    )