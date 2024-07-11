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
from langchain_openai import ChatOpenAI

TOP_K = 5

username = os.getlogin()
if username == "andrea":
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
else:
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
os.environ["GROQ_API_KEY"] = "KEY"

qa_df = pd.read_pickle("data/ragas_qa.pickle").head(10)
print(qa_df)


def evaluation(qa_df, formatted_time):
    """
    Perform evaluation on the given QA dataframe using the specified models.

    Args:
        qa_df (pandas.DataFrame): The QA dataframe containing questions and answers.
        formatted_time (str): The formatted time to be used in the output file name.

    Returns:
        None
    """
    chatllm = ChatOllama(model="llama3", base_url=os.environ.get("OLLAMA_BASE_URL"))
    # chatllm = ChatGroq(model="llama3")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    try:
        results = Evaluate(qa_df, chatllm, embeddings)
    finally:
        with open(
            f"eval_results/pipeline_result_{formatted_time}.json", "w"
        ) as outfile:
            data = {"RESULT": results, "CONFIG": config}
            json.dump(data, outfile, indent=4)


def main(config):
    """
    Run the RAGPipeline to generate answers for a set of questions.

    Args:
        config (dict): Configuration parameters for the RAGPipeline.

    Returns:
        None
    """
    global qa_df
    experiment = input(
        "What is the subject/component this evaluation is concerned about?: "
    )
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
        chat_mode=False,
    )

    # Use tqdm for progress bar
    generated_responses = []
    contexts = []
    meta_data = []
    scores = []

    # Use tqdm for progress bar
    for question in tqdm(
        qa_df["question"], desc="Processing questions", unit="question"
    ):
        generated, context, _ = pipeline.run(question)
        pprint(context)
        generated_responses.append(generated)
        context_texts = [dict1["text"] for dict1 in context]
        contexts.append(json.dumps(context_texts))  # Convert list to JSON string
        meta_data.append(
            json.dumps([dict1["metadata"] for dict1 in context])
        )  # Convert list to JSON string
        scores.append(
            json.dumps([dict1["score"] for dict1 in context])
        )  # Convert list to JSON string

    # Add the generated responses and contexts to the DataFrame
    qa_df["answer"] = generated_responses
    qa_df["contexts"] = contexts
    qa_df["metadata"] = meta_data
    qa_df["score"] = scores

    # Save the updated DataFrame to a new CSV file
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    if not os.path.exists("results"):
        os.mkdir("results")
    qa_df.to_csv(f"results/pipeline_{formatted_time}_{experiment}.csv", index=False)
    # Convert JSON strings back to Python lists
    qa_df["contexts"] = qa_df["contexts"].apply(json.loads)
    qa_df["metadata"] = qa_df["metadata"].apply(json.loads)
    qa_df["score"] = qa_df["score"].apply(json.loads)
    evaluation(qa_df, formatted_time)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    main(config)
