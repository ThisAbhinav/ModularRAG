from abc import ABC, abstractmethod, abstractproperty
from utils import loadllm
from llama_index.core.llms import ChatMessage
import json
import re
import pandas as pd
from tqdm import tqdm
from logger_config import logger  # Assuming you have this setup
import os

username = os.getlogin()
if username == "andrea":
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
else:
    os.environ["OLLAMA_BASE_URL"] = "http://10.129.152.197:11434"
os.environ["GROQ_API_KEY"] = "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"


llm = loadllm("Ollama", temperature=0)


def context_precision(query, ground_truth, retrieved_contexts_combined, retrieved_contexts, retries=3):
    retrieved_contexts_list = json.loads(retrieved_contexts)

    logger.info(f"Evaluating context precision for query: {query}")

    system_prompt = """
    You are an expert text relevancy evaluator. Given a question, the ground truth, and a list of passages 
    (retrieved contexts) along with their indices, analyze each sentence by comparing it with the ground truth,
    and evaluate its relevance to the question.
    """

    prompt = """
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.

    Given the question:
    {query}

    Here is the ground truth:
    {ground_truth}

    Here are the retrieved contexts with their respective indices:
    {retrieved_contexts_combined}
    
    Follow these steps one by one:

    A. Evaluate each context in the retrieved contexts list for relevance to the question based on the ground truth.
    B. For each relevant context, note its index which has been given along with it, and strictly make sure it is within the range (0, {max_index})
    C. Collect all indices of relevant contexts into a list, please do not append any index that is not in the specified range.

    Return the results in the following JSON format:
    {{
        "relevant_indices": [...]
    }}
    
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.

    """

    for attempt in range(retries):
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt.format(query=query, retrieved_contexts_combined=retrieved_contexts_combined,
                                                           ground_truth=ground_truth, max_index=len(retrieved_contexts_list)-1)),
        ])
        print(prompt.format(query=query, retrieved_contexts_combined=retrieved_contexts_combined,
                                                           ground_truth=ground_truth, max_index=len(retrieved_contexts_list)-1))
        try:
            result = json.loads(str(response.message.content))
            break
        except json.JSONDecodeError:
            result = extract_json_manually(str(response.message.content))
            if result:
                break
    else:
        logger.error(f"Failed to parse response as JSON after {retries} attempts.")
        return 0

    relevant_indices = result.get("relevant_indices", [])
    print(relevant_indices)
    if not relevant_indices:
        cumulative_precision = 0
    else:
        retrieved_contexts_list = json.loads(retrieved_contexts)
        relevant_nodes = 0
        sum_precision = 0
        for i in range(len(retrieved_contexts_list)):
            k = i + 1
            if i in relevant_indices:
                relevant_nodes += 1
                sum_precision += (relevant_nodes / k)
        cumulative_precision = sum_precision / len(relevant_indices)

    logger.info(f"Cumulative precision for query '{query}': {cumulative_precision}")
    return cumulative_precision


def extract_json_manually(text):
    try:
        match = re.search(r'\{.*?}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None


def evaluate_context_precision(csv_path):
    df = pd.read_csv(csv_path)
    precision_scores = []

    logger.info(f"Evaluating context precision for {len(df)} queries...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = row["question"]
        retrieved_contexts = row["contexts"]
        ground_truth = row["ground_truth"]
        combined_text = ""
        retrieved_contexts_list = json.loads(retrieved_contexts)
        for i, passage in enumerate(retrieved_contexts_list):
            passage = passage.replace("\n", " ")
            combined_text += f"INDEX {i}) +++++{passage}\n\n"

        precision = context_precision(query, ground_truth, combined_text, retrieved_contexts)
        try:
            precision_scores.append(precision)
        except Exception as e:
            logger.error(f"Error processing row '{index}': {e}")

    average_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    logger.info(f"Average Precision: {average_precision}")
    return average_precision


csv_path = "/home/deepakachu/Desktop/RAG/26_Assessment_RAG/results/pipeline_2024-06-19_01-05_recbaseprecision.csv"
average_precision = evaluate_context_precision(csv_path)
print(f"Average Precision: {average_precision}")
