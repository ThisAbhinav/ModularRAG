from abc import ABC, abstractmethod, abstractproperty
from utils import loadllm
from llama_index.core.llms import ChatMessage
import json
import re
import pandas as pd
from tqdm import tqdm
import logging
from logger_config import logger  # Assuming you have this setup
import os

# Setup environment variables based on the username
username = os.getlogin()
if username == "andrea":
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
else:
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
os.environ["GROQ_API_KEY"] = "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"

# Load the LLM
llm = loadllm("Ollama", temperature=0)

