from llama_index.core.postprocessor import SentenceEmbeddingOptimizer, SentenceTransformerRerank, SimilarityPostprocessor


def get_SentenceEmbeddingOptimizer() -> SentenceEmbeddingOptimizer:
    postprocessor = SentenceEmbeddingOptimizer(
        percentile_cutoff=0.3,
        # threshold_cutoff=0.7
    )
    return postprocessor

def get_SimilarityPostprocessor() -> SimilarityPostprocessor:
    return SimilarityPostprocessor(similarity_cutoff=0.3)

def get_SentenceTransformerRerank() -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=5
        )

