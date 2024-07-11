import chainlit as cl
from chunking import (
    RecursiveChunking,
    SemanticChunking,
    SentenceChunking,
    HierarchicalChunking,
    SentenceWindowChunking,
    AgenticChunking,
)
from pprint import pprint
from vectorstore import VectorStore
from retriever import BaseRetriever
from logger_config import logger
from utils import loadEmbeddingModel, loadllm
from RAGpipeline import RAGPipeline
from generation import SimpleGenerator
import os
import pandas as pd
from tqdm import tqdm
from evaluator import Evaluate
from datasets import Dataset
import datetime
from ast import literal_eval
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
import json
import asyncio
from querying import BaseQuery, HyDE
from reranking import CrossEncoderReranker, PairWiseReRanker, ListWiseReranker

message_history = []
username = os.getlogin()
if username == "andrea":
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
else:
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
os.environ["GROQ_API_KEY"] = "KEY"

with open("config.json", "r") as file:
    config = json.load(file)
get = globals()
datatype = config["datatype"]
if datatype == "markdown":
    datapath = "outputs_md"
elif datatype == "pdf":
    datapath = "data/themebooks"
else:
    raise ValueError("Invalid datatype")
chunker = get[config["chunker"]]()
embed_model = loadEmbeddingModel(
    config["embed_model"]["type"], config["embed_model"]["name"]
)
llm = loadllm(config["generator"]["name"])
query_method = get[config["query_method"]]()
vectorstore = VectorStore(name=config["vector_store_name"])
retriever = get[config["retriever"]["type"]](config["retriever"]["top_k"])
reranker_type = config["reranker"]["type"]
reranker_top_k = config["reranker"]["top_k"]
reranker_model_name = config["reranker"]["model_name"]
reranker = get[reranker_type](llm, reranker_top_k, reranker_model_name)
generator = get[config["generator"]["type"]](llm)

pipeline = RAGPipeline(
    generator=generator,
    vector_store=vectorstore,
    chunker=chunker,
    retriever_type=retriever,
    embed_model=embed_model,
    datatype=datatype,
    datapath=datapath,
    query_method=query_method,
    reranker=reranker,
    rerank=config["rerank"],
    use_cache=config["use_cache"],
    use_router=config["use_router"],
    chat_mode=True,
    chat_history=[],
)


@cl.on_chat_start
async def chat_start():
    msg = cl.Message(content="Hello! I'm Eyantra Chatbot. You can ask me any queries!")
    await msg.send()


@cl.on_message
async def main(question):
    response, context, chat_history = pipeline.run(question.content)
    # Format the response and context for Chainlit
    answer_text = f"**Answer:** {response}\n\n"
    context_text = "**Retrieved Context:**\n"
    for idx, ctx in enumerate(context, 1):
        context_text += f"{idx}. {ctx}\n"

    final_response = answer_text + context_text
    print("message_history: ", chat_history)
    await cl.Message(final_response).send()
