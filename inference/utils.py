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
import re
from llama_index.core import Document
import json
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
    "VD20.pdf": "Vitarana Drone",
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
os.environ["GROQ_API_KEY"] = "KEY"


def getllamaIndexDocument(dataframe):
    """
    Converts a dataframe into a list of Document objects.

    Args:
        dataframe (pandas.DataFrame): The input dataframe containing 'text' and 'metadata' columns.

    Returns:
        list: A list of Document objects, where each Document object represents a row in the dataframe.
    """
    documents = []
    for index, row in dataframe.iterrows():
        text = row["text"]
        metadata = row["metadata"]
        document = Document(text=text, metadata=metadata)
        documents.append(document)
    return documents


def loadEmbeddingModel(host, name=None):
    """
    Load an embedding model based on the specified host.

    Args:
        host (str): The host of the embedding model. Supported values are "ollama" and "huggingface".
        name (str, optional): The name of the model to load. If not provided, a default model will be used.

    Returns:
        model: The loaded embedding model.

    Raises:
        ValueError: If an unsupported model type is specified.

    """
    if host == "ollama":
        default_name = "nomic-embed-text"
        (
            logger.info(
                f"Loading ollama embedding model {default_name} by default. You can change it with the 'name' parameter."
            )
            if name is None
            else logger.info(f"Loading ollama model {name}")
        )
        try:
            model = OllamaEmbedding(
                model_name=name or default_name,
                base_url=os.environ.get("OLLAMA_BASE_URL"),
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
    """
    Load a language model based on the specified host.

    Args:
        host (str): The host name of the language model. Supported values are "Ollama", "Groq", and "ChatOllama".
        name (str, optional): The name of the language model to load. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to DEFAULT_MAX_TOKENS.
        temperature (float, optional): The temperature for generating text. Defaults to DEFAULT_TEMP.

    Returns:
        object: The loaded language model object.

    Raises:
        ValueError: If an unsupported model type is specified.

    """
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
    elif host == "OpenAI":

        # api_key = os.environ.get("OPENAI_API_KEY")
        api_key = "KEY"
        print(api_key)

        if not api_key:
            logger.error(
                "OPENAI_API_KEY environment variable not set. Please set it to your OPEN AI API key."
            )
            return None
        default_name = "gpt-4o"
        try:
            if name is None:
                model = OpenAI(model=default_name, api_key=api_key)
            else:
                model = OpenAI(model=name, api_key=api_key)
            return model
        except Exception as e:
            logger.error(f"Error loading Open AI llm model: {e}")
    elif host == "ChatOllama":
        default_name = "llama3"
        (
            logger.info(
                f"Loading ollama llm {default_name} by default. You can change it with the 'name' parameter."
            )
            if name is None
            else logger.info(f"Loading ollama llm {name}")
        )
        try:
            model = ChatOllama(
                model=name or default_name,
                base_url=os.environ.get("OLLAMA_BASE_URL"),
                max_tokens=2000,
            )
            return model
        except Exception as e:
            logger.error(f"Error loading ChatOllama model: {e}")
    else:
        logger.error(f"Unsupported model type: {host}")
        raise ValueError(f"Unsupported model type: {host}")


def getContextString(nodes):
    """
    Returns a string containing the text of each node in the given list.

    Args:
        nodes (list): A list of nodes.

    Returns:
        str: A string containing the text of each node, separated by newlines.
    """
    contexts = []
    for node in nodes:
        contexts.append(node.text)
    context_str = "\n".join(contexts)
    return context_str


def addMetadata(dict1, input_type):
    """
    Adds metadata to the given dictionary based on the input type.

    Parameters:
    - dict1 (dict): The dictionary to which metadata will be added.
    - input_type (str): The type of input (e.g., "pdf").

    Returns:
    - dict: The updated dictionary with added metadata.
    """
    if input_type == "pdf":
        name = dict1["source"].split(os.path.sep)[-1]
    else:
        name = dict1["source"].split(os.path.sep)[1] + ".pdf"
    if name in themedict:
        dict1["theme"] = themedict[name]
        dict1["year"] = yeardict[name]
    return dict1


def nodeExtractor(context, datatype):
    """
    Extracts nodes from the given context and returns a list of dictionaries containing node information.

    Parameters:
    - context (list): The context from which nodes are to be extracted.
    - datatype (str): The datatype of the nodes.

    Returns:
    - context_nodes (list): A list of dictionaries containing node information. Each dictionary has the following keys:
        - "text" (str): The text of the node.
        - "metadata" (dict): The metadata of the node.
        - "score" (float): The score of the node.

    """
    context_nodes = []
    for node in context:
        context_nodes.append(
            {
                "text": f"{node.text}",
                "metadata": node.metadata,
                "score": node.score,
            }
        )
    return context_nodes


def getContextStringSentenceWindow(nodes):
    """
    Returns a string representation of the window metadata for each node in the given list.

    Args:
        nodes (list): A list of nodes.

    Returns:
        str: A string representation of the window metadata for each node, separated by newlines.
    """
    context = []
    for node in nodes:
        context.append(node.metadata["window"])
    context_str = "\n".join(context)
    return context_str


def messages_to_history_str(messages) -> str:
    """Converts a list of messages to a history string.

    Args:
        messages (list): A list of messages.

    Returns:
        str: A string representing the history of messages.

    """
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"
        string_messages.append(string_message)
    return "\n".join(string_messages)


def append_neighbors_conditional(documents, max_length=500):
    """
    Append parts of the previous and succeeding documents to each document until it reaches max_length.
    Equal proportions from previous and succeeding documents are added, except for the first and last documents.
    
    Parameters:
    documents (list of str): List of documents.
    max_length (int): Maximum length for each document.
    
    Returns:
    list of str: List of documents with appended parts.
    """
    modified_documents = []
    num_docs = len(documents)
    
    for i in range(num_docs):
        current_doc = documents[i].text
        
        # If the current document is already longer than max_length, keep it as is
        if len(current_doc) >= max_length:
            llama_index_doc = Document()
            llama_index_doc.text = current_doc
            llama_index_doc.metadata = documents[i].metadata
            modified_documents.append(llama_index_doc)
            continue
        
        # Calculate the required length to add
        remaining_length = max_length - len(current_doc)
        
        # Initialize parts to be added
        prev_doc_part = ""
        next_doc_part = ""
        
        # Add parts from the previous documents
        if i > 0:
            j = i - 1
            while j >= 0 and len(prev_doc_part) < remaining_length // 2:
                prev_doc_part = documents[j].text[-((remaining_length // 2) - len(prev_doc_part)):] + prev_doc_part
                j -= 1
        
        # Add parts from the succeeding documents
        if i < num_docs - 1:
            j = i + 1
            while j < num_docs and len(next_doc_part) < remaining_length // 2:
                next_doc_part += documents[j].text[:remaining_length // 2 - len(next_doc_part)]
                j += 1
        
        # Combine parts
        new_doc = prev_doc_part + current_doc + next_doc_part
        llama_index_doc = Document()
        llama_index_doc.text = new_doc
        llama_index_doc.metadata = documents[i].metadata
        modified_documents.append(llama_index_doc)
    
    return modified_documents

def updateHash(llama_index_nodes):
    """
    Update the hash values of the given list of LlamaIndexNode objects.

    Args:
        llama_index_nodes (list): A list of LlamaIndexNode objects.

    Returns:
        list: The updated list of LlamaIndexNode objects with updated hash values.
    """
    directory = 'data'
    file_name = 'hashes.json'
    file_path = os.path.join(directory, file_name)
    os.makedirs(directory, exist_ok=True)
    # Create the JSON content
    json_content = {
        "hash": 0
    }

    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(json_content, file, indent=4)

    with open(file_path, 'r') as file:
        data = json.load(file)
    for node in llama_index_nodes:
        data["hash"] += 1
        node.id_ = data["hash"]
        
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    return llama_index_nodes