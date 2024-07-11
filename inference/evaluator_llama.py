from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    context_entity_recall,
    answer_similarity,
)
from datasets import Dataset
from ast import literal_eval

from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
import pandas as pd

pd.set_option("display.max_colwidth", 0)
import nest_asyncio

nest_asyncio.apply()
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response


def get_eval_results(key, eval_results):
    """
    Calculate and return the score for a given key in the eval_results dictionary.

    Parameters:
    key (str): The key to retrieve the results from the eval_results dictionary.
    eval_results (dict): A dictionary containing evaluation results for different keys.

    Returns:
    float: The score calculated based on the number of correct results divided by the total number of results.

    """
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


async def Evaluate(df, llm, embeddings):
    """
    Evaluate the given dataframe using the specified language model (llm) and embeddings.

    Args:
        df (pandas.DataFrame): The dataframe to be evaluated.
        llm: The language model used for evaluation.
        embeddings: The embeddings used for evaluation.

    Returns:
        None
    """
    df["contexts"] = df["contexts"].apply(lambda x: literal_eval(str(x)))
    df["real_contexts"] = df["real_contexts"].apply(lambda x: literal_eval(str(x)))

    faithfulness_llm = FaithfulnessEvaluator(llm=llm)
    relevancy_llm = RelevancyEvaluator(llm=llm)
    correctness_llm = CorrectnessEvaluator(llm=llm)

    documents = SimpleDirectoryReader("./data/themebooks").load_data()
    splitter = SentenceSplitter(chunk_size=512)
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_llm, "relevancy": relevancy_llm},
        workers=8,
    )
    vector_index = VectorStoreIndex.from_documents(
        documents, transformations=[splitter]
    )
    eval_results = await runner.aevaluate_queries(
        vector_index.as_query_engine(llm=llm), queries=df["question"]
    )

    get_eval_results("faithfulness", eval_results)
    get_eval_results("relevancy", eval_results)
    get_eval_results("relevancy", eval_results)
    # return result
