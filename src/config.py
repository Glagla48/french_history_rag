import sys
from pathlib import Path

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = Path(PROJECT_ROOT / "data" / "raw" / "french")
PERSIST_DIR = Path(PROJECT_ROOT / "data" / "chroma_db")
CLEAN_DATA_DIR = Path(PROJECT_ROOT / "data" / "clean" / "french")

OLLAMA_LLM_MODEL = "llama3.2:latest"
OLLAMA_EMBED_MODEL = "embeddinggemma"
OLLAMA_REQUEST_TIMEOUT = 600.0
OLLAMA_MAX_CONCURRENT_DOCS = 2

SYSTEM_PROMPT_PATH = Path(PROJECT_ROOT / "src" /'rag' / "util" / "system_prompt.txt")
CONTEXT_PROMPT_PATH = Path(PROJECT_ROOT / "src" / 'rag'/"util"/ "context_prompt.txt")


def get_ollama_llm(**kwargs) -> Ollama:
    return Ollama(
        model=kwargs.pop("model", OLLAMA_LLM_MODEL),
        request_timeout=kwargs.pop("request_timeout", OLLAMA_REQUEST_TIMEOUT),
        temperature=kwargs.pop("temperature", 0.1),
        keep_alive="10m",
        **kwargs,
    )


def get_ollama_embed_model(**kwargs) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=kwargs.pop("model_name", OLLAMA_EMBED_MODEL),
        request_timeout=kwargs.pop("request_timeout", OLLAMA_REQUEST_TIMEOUT),
        keep_alive="10m",
        **kwargs,
    )
