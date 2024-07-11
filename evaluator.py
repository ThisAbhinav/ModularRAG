import pandas as pd
from ast import literal_eval
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


def convert_to_list(df, column_name):
    # Convert string representations of lists to actual lists
    df[column_name] = df[column_name].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )
    # Ensure the column is a list of strings
    df[column_name] = df[column_name].apply(lambda x: x if isinstance(x, list) else [])


def Evaluate(df, llm, embeddings):
    # Convert the necessary columns to lists of strings
    convert_to_list(df, "contexts")
    convert_to_list(df, "real_contexts")

    # Drop unnecessary columns
    new_df = df.drop(
        [
            "qid",
            "real_contexts",
            "meta_data_real_contexts",
            "evolution_type",
            "metadata",
            "score",
        ],
        axis=1,
    )

    # Create the Dataset object
    dataset = Dataset.from_pandas(new_df)

    print(dataset)

    # Evaluate the dataset
    result = evaluate(
        dataset,
        metrics=[
            # answer_relevancy,
            # faithfulness,
            context_recall,
            context_precision,
            context_relevancy,
            # answer_correctness,
            # context_entity_recall,
            # answer_similarity,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    return result
