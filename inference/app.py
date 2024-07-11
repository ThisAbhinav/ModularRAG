import chainlit as cl
from chunking import (
    RecursiveChunking,
    SemanticChunking,
    SentenceChunking,
    HierarchicalChunking,
    SentenceWindowChunking,
    AgenticChunking,
    MarkdownChunking,
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
from dataingestion import MarkDownDataIngestion, LLMSherpaIngestion
import shutil
username="andrea"
#username = os.getlogin()
if username == "andrea":
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
else:
    # os.environ["OLLAMA_BASE_URL"] = "http://10.195.100.5:11434"
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
ingestor = get[config["ingestor"]]()
chunker = get[config["chunker"]]()
embed_model = loadEmbeddingModel(
    config["embed_model"]["type"], config["embed_model"]["name"]
)
llm = loadllm(host = config["generator"]["name"],name = config["generator"]["model_name"])
query_method = get[config["query_method"]]()
vectorstore = VectorStore(name=config["vector_store_name"])
retriever = get[config["retriever"]["type"]](config["retriever"]["top_k"])
reranker_type = config["reranker"]["type"]
reranker_top_k = config["reranker"]["top_k"]
reranker_model_name = config["reranker"]["model_name"]
reranker = get[reranker_type](llm, reranker_top_k, reranker_model_name)
generator = get[config["generator"]["type"]](llm)

global pipeline
pipeline = RAGPipeline(
    generator=generator,
    vector_store=vectorstore,
    chunker=chunker,
    ingestor=ingestor,
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


@cl.on_chat_end
def on_chat_end():
    global pipeline

    pipeline = RAGPipeline(
        generator=generator,
        vector_store=vectorstore,
        chunker=chunker,
        ingestor=ingestor,
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
    """
    Callback function called when the chat starts.
    Sends a greeting message to the user.
    """
    msg = cl.Message(content="Hello! I'm Eyantra Chatbot. You can ask me any queries!")

    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    """
    Main function that handles incoming messages from the user.
    Runs the RAGPipeline to generate a response based on the user's question.
    Sends the response and retrieved context back to the user.
    """
    if len(message.elements)!= 0:
        await cl.Message(
        content=f"Received: {message.elements[0].name} \n Adding the file to the corpus..."
        ).send()
        await cl.Message(
        content=f"This chat will now be in file mode. You can ask questions specifically related to the file. Please refresh the page or start a new chat to ask question from the full corpus."
        ).send()
        global pipeline
        path = message.elements[0].path.replace("\\", "/").replace("C:/Users/abhin/Downloads/projects/Eysip_Rag_Project/26_Assessment_RAG/", "").replace("/app/inference/", "").replace("inference/", "")
        file_name = os.path.basename(path)
        print("PATH : ",path)
        destination_path = os.path.join("data/themebooks", file_name)
        shutil.copy(path, destination_path)
        print(f"File copied from temporary file to data/themebooks")
        path = destination_path
        pipeline.add_file(path)


    response, context, chat_history = pipeline.run(message.content)
    # Format the response and context for Chainlit
    answer_text = f"**Answer:** {response}\n\n"
    context_text = "**Retrieved Context:**\n"
    context = context[:5]
    for idx, ctx in enumerate(context, 1):
        location = ctx['metadata']['source'].replace('\\', '/')
        context_text += f"{idx}. {ctx['text'][0:75]}... \n [Reference](http://10.129.152.197:8001/{location}#page={ctx['metadata']['page_number']}) \n \n"
    final_response = answer_text + context_text
    print("message_history: ", chat_history)
    await cl.Message(final_response).send()
