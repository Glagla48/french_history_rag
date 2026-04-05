# french_history_rag
This is RAG in a webapp answering questions about French History.

# Installation
## Python 
Download Python 3.13.5 [here](https://www.python.org/downloads/) and follow the instructions given by the installation setup.

## Download Ollama and llm
Download ollama [here](https://ollama.com/) and follow the instructions given by the installation setup.\n
Follow the instructions [here](https://ollama.com/library/embeddinggemma) to download the embedding model.
Follow the instructions [here](https://ollama.com/library/llama3.2) to download the chat model.

## Requirments
To download the librairies use:
```Bash
pip -r install requirments.txt
```

## Folder initilisation
To create the data fodlers, use:
```Bash
./folder_init.sh
```

# Scrap Wikipedia
```Bash
pip -r install requirments.txt
```

# Activate your RAG !
```Bash
streamlit run src/app.py
```