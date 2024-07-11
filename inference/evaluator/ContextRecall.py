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


def context_recall(ground_truth, retrieved_contexts, retries=3):
    try:
        retrieved_contexts_list = json.loads(retrieved_contexts)
        context_string = ""
        for context in retrieved_contexts_list:
            context = context.replace("\n", " ")
            context_string += context + "\n"

        statements = get_statements(ground_truth)

        ground_truth_string = ""
        for i, statement in enumerate(statements):
            ground_truth_string += f"INDEX{i}) {statement}\n"

        system_prompt = """
        You are an expert relevancy evaluator. Given a list of statements, and a context passage, analyse which statements can be attributed to the context passage
        """

        prompt = """
        **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.

        Given the list of statements with their indices:
        {ground_truth_string}

        Here is the context passage:
        <PASSAGE STARTS>
        {context_string}
        <PASSAGE ENDS>

        Follow these steps one by one:
        A. For each statement in the list of statements, evaluate if that statement can be attributed to the context passage
        B. For each attributable statement, note down its index which has been provided, STRICTLY make sure it is within the range (0, {max_index})
        C. Add all the indices of attributable statements to a list

        Return the list in the following format:
        {{
            "attributable_indices": [...]
        }}

        **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.
        """

        for attempt in range(retries):
            response = llm.chat([
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=prompt.format(ground_truth_string=ground_truth_string,
                                                               context_string=context_string,
                                                               max_index=len(statements) - 1)),
            ])

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

        attributable_indices = result.get("attributable_indices", 0)

        try:
            if not attributable_indices:
                recall_score = 0
            else:
                recall_score = len(attributable_indices) / len(statements)
        except ZeroDivisionError:
            logger.error("Division by zero error in recall score calculation.")
            recall_score = 0
        logger.info(f"Recall score for '{ground_truth[:30]}...': {recall_score}")
        return recall_score

    except Exception as e:
        logger.error(f"An error occurred during context recall calculation: {e}")
        return 0


def get_statements(passage, retries=3):
    system_prompt = """
    You are an expert AI that can extract individual statements from a given piece of text.
    """

    prompt = """
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.

    Given the text passage:
    <PASSAGE STARTS>
    {passage}
    <PASSAGE ENDS>
    ####################################################################################################################
    Follow these steps one by one:
    A. Analyse the passage and extract all the individual, disjoint statements.
    B. Put all the extracted statements into a list 

    Return the results in the following JSON format:
    {{
        "statements": [<statement 1>, <statement 2>, ...]
    }}
    ####################################################################################################################

    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.
    """

    for attempt in range(retries):
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt.format(passage=passage))
        ])

        try:
            result = json.loads(str(response.message.content))
            break
        except json.JSONDecodeError:
            result = extract_json_manually(str(response.message.content))
            if result:
                break
    else:
        logger.error(f"Failed to parse response as JSON after {retries} attempts.")
        return []

    return result.get("statements", [])


def extract_json_manually(text):
    try:
        match = re.search(r'\{.*?}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None


def calculate_aggregate_score(csv_path):
    logger.info(f"Reading CSV file from path: {csv_path}")
    df = pd.read_csv(csv_path)
    scores = []

    logger.info(f"Starting to process {len(df)} rows from the CSV file.")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        retrieved_contexts = row["contexts"]
        ground_truth = row["ground_truth"]
        logger.debug(
            f"Processing row {index}: ground_truth='{ground_truth[:30]}...', retrieved_contexts='{retrieved_contexts[:30]}...'")
        try:
            score = context_recall(ground_truth, retrieved_contexts)
            scores.append(score)
        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")

    if scores:
        avg_recall_score = sum(scores) / len(scores)
        logger.info(f"Calculated aggregate recall score: {avg_recall_score}")
    else:
        avg_recall_score = 0
        logger.warning("No valid scores found. Aggregate recall score is set to 0.")

    return avg_recall_score


csv_path = "/home/deepakachu/Desktop/RAG/26_Assessment_RAG/results/pipeline_2024-06-19_01-01_rechydeprecision.csv"
aggregate_score = calculate_aggregate_score(csv_path)
print(f"Aggregate Score: {aggregate_score}")


