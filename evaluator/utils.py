import pandas as pd
import os
from logger_config import logger
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

themedict = {
    "AB20.pdf": "Agri Bot",
    "AB23.pdf": "Astro Bot",
    "CL23.pdf": "Cosmo Logistic",
    "FR22.pdf": "Functional Weeder",
    "GG23.pdf": "Geo Guide",
    "HB23.pdf": "Hologlyph Bots",
    "LD23.pdf": "Luminosity Drone",
    "SB20.pdf": "Sahayak Bot",
    "SS21.pdf": "Vitaraṇa Drone",
    "VB20.pdf": "Vargi Bots",
    "VD20.pdf": "Vitaraṇa Drone",
}

yeardict = {
    "AB20.pdf": 2020,
    "AB23.pdf": 2023,
    "CL23.pdf": 2023,
    "FR22.pdf": 2022,
    "GG23.pdf": 2023,
    "HB23.pdf": 2023,
    "LD23.pdf": 2023,
    "SB20.pdf": 2020,
    "SS21.pdf": 2021,
    "VB20.pdf": 2020,
    "VD20.pdf": 2020,
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMP = 0.6
os.environ["GROQ_API_KEY"] = (
    "key"
)

def getllamaIndexDocument(dataframe):
    documents = []
    for index, row in dataframe.iterrows():
        text = row["text"]
        metadata = row["metadata"]
        document = Document(text=text, metadata=metadata)
        documents.append(document)
    return documents


def loadEmbeddingModel(host, name=None):
    if host == "ollama":
        default_name = "nomic-embed-text"
        (
            logger.info(
                f"Loading ollama model {default_name} by default. You can change it with the 'name' parameter."
            )
            if name is None
            else logger.info(f"Loading ollama model {name}")
        )
        try:
            model = OllamaEmbedding(
                model_name=name or default_name, base_url=os.environ.get("OLLAMA_BASE_URL")
            )
            return model
        except Exception as e:
            logger.error(f"Error loading ollama model: {e}")
    elif host == "huggingface":
        default_name = "sentence-transformers/all-mpnet-base-v2"
        # another good model:
        (
            logger.info(
                f"Loading huggingface model {default_name} by default. You can change it with the 'name' parameter."
            )
            if name is None
            else logger.info(f"Loading huggingface model {name}")
        )
        try:
            model = HuggingFaceEmbedding(model_name=name or default_name)
            return model
        except Exception as e:
            logger.error(f"Error loading huggingface model: {e}")
    else:
        raise ValueError(f"Unsupported model type: {host}")


def loadllm(host, name=None, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMP):
    if host == "Ollama":
        default_name = "llama3"
        (
            logger.info(
                f"Loading ollama llm {default_name} by default. You can change it with the 'name' parameter."
            )
            if name is None
            else logger.info(f"Loading ollama llm {name}")
        )
        try:
            Settings.llm = Ollama(
                model=name or default_name,
                base_url=os.environ.get("OLLAMA_BASE_URL"),
                request_timeout=60.0,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return Settings.llm
        except Exception as e:
            logger.error(f"Error loading ollama llm model: {e}")
    # elif host == "ChatGroq":
    #     api_key = os.environ.get("GROQ_API_KEY")
    #     if not api_key:
    #         logger.error(
    #             "GROQ_API_KEY environment variable not set. Please set it to your Groq API key."
    #         )
    #         return None
    #     default_name = "llama3-8b-8192"
    #     try:
    #         model = ChatGroq(model_name=default_name)
    #         return model
    #     except Exception as e:
    #         logger.error(f"Error loading groq llm model: {e}")
    elif host == "Groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error(
                "GROQ_API_KEY environment variable not set. Please set it to your Groq API key."
            )
            return None
        default_name = "llama3-8b-8192"
        try:
            model = Groq(model=default_name, api_key=api_key)
            return model
        except Exception as e:
            logger.error(f"Error loading groq llm model: {e}")
    # elif host == "ChatOllama":
    #     default_name = "llama3"
    #     (
    #         logger.info(
    #             f"Loading ollama llm {default_name} by default. You can change it with the 'name' parameter."
    #         )
    #         if name is None
    #         else logger.info(f"Loading ollama llm {name}")
    #     )
    #     try:
    #         model = ChatOllama(
    #             model=name or default_name, base_url=BASE_OLLAMA_URL, max_tokens=2000
    #         )
    #         return model
    #     except Exception as e:
    #         logger.error(f"Error loading ChatOllama model: {e}")
    else:
        logger.error(f"Unsupported model type: {host}")
        raise ValueError(f"Unsupported model type: {host}")


def getContextString(nodes):
    contexts = []
    for node in nodes:
        contexts.append(node.text)
    context_str = "\n".join(contexts)
    return context_str


def addMetadata(dict1, input_type):
    if input_type == "pdf":
        name = dict1["source"][-8:]
    else:
        name = dict1["source"].split("/")[1] + ".pdf"
    dict1["theme"] = themedict[name]
    dict1["year"] = yeardict[name]
    return dict1


def nodeExtractor(context, datatype):
    context_nodes = []
    for node in context:
        context_nodes.append(
            {
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score,
            }
        )
    return context_nodes


def getContextStringSentenceWindow(nodes):
    context = []
    for node in nodes:
        context.append(node.metadata["window"])
    context_str = "\n".join(context)
    return context_str
