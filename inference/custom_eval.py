import json
import re
import llama_cpp
from utils import loadllm
from llama_index.core.llms import ChatMessage

# Assuming loadllm correctly initializes and returns a language model instance
llm = loadllm("Groq", temperature=0)


def extract_json_manually(text):
    """
    Extracts a JSON object from the given text.

    Args:
        text (str): The text containing the JSON object.

    Returns:
        dict or None: The extracted JSON object as a dictionary, or None if no JSON object is found.
    """
    try:
        match = re.search(r'\{.*?}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None


def split_into_statements(text):
    """
    Splits the given text into individual statements.

    Args:
        text (str): The text to be split into statements.

    Returns:
        list: A list of individual statements.

    """
    return [sentence.strip() for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?])\s', text) if
            sentence.strip()]


def answer_relevancy(answer, ground_truth, query, retries=3):
    """
    Evaluate the relevancy of an answer based on the ground truth and the query.

    Args:
        answer (str): The answer to be evaluated.
        ground_truth (str): The ground truth against which the answer is compared.
        query (str): The query for which the answer is provided.
        retries (int, optional): The number of retries to parse the response as JSON. Defaults to 3.

    Returns:
        None

    Raises:
        None

    Example Usage:
        answer_relevancy(answer, ground_truth, query)
    """
    system_prompt = """You are an expert text relevancy evaluator. Given a question, an answer, and the ground truth,
    analyze the answer by comparing it with the ground truth and the question.
    """



    prompt = """
    Given the question: 
    {query}

    Here is the answer:
    {answer}

    Here is the ground truth:
    {ground_truth}

    For the given question, analyze whether the provided answer is correct based on the ground truth. 
    Then follow these steps:

    1. Extract all the relevant statements from the provided answer, not the ground truth.
    2. Put the relevant statements in a list.

    Return the results in the following JSON format:
    {
        "relevant_statements": [...]
    }

    Return only the JSON formatted results without any surrounding text.
    
    Example:
    
    Question: 
    What is the execution time for the task involving picking and placing red tomatoes with perception?

    Answer:
    The execution time for the task involving picking and placing red tomatoes with perception is not explicitly stated in the given context. However, it is mentioned that the execution time is the interval between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement.

    Ground Truth:
    The execution time is the interval between the first time the robots get spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement.

    Example JSON:
    {
        "relevant_statements": [
            "The execution time for the task involving picking and placing red tomatoes with perception is not explicitly stated in the given context.",
            "However, it is mentioned that the execution time is the interval between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement."
        ]
    }
    """

    for attempt in range(retries):
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt)
        ])

        try:
            result = json.loads(str(response.message.content))
            break
        except json.JSONDecodeError:
            result = extract_json_manually(str(response.message.content))
            if result:
                break
    else:
        print(f"Failed to parse response as JSON after {retries} attempts.")
        print("Response:", response.message.content)
        return

    try:
        relevant_statements = result["relevant_statements"]
        all_statements = split_into_statements(answer)
        num_correct = len(relevant_statements)
        num_total = len(all_statements)
        relevancy_score = num_correct / num_total if num_total > 0 else 0
        print(f"Relevancy Score: {relevancy_score:.2f}")
        print("Relevant Statements:")
        print(json.dumps(relevant_statements, indent=4))
    except (KeyError, TypeError):
        print("Error in the structure of the returned JSON.")
        print("Response:", result)


def faithfulness(query, answer, retrieved_context):
    system_prompt = """You are an expert text factual correctness evaluator. Given a question, an answer, 
    and the retrieved context from which the answer is generated,
    analyze the answer with respect to the retrieved context and evaluate its factual correctness.
    The retrieved context is to be treated as undisputed, true facts.
    """

    prompt = """
    Given the question:
    {query}

    Here is the answer:
    {answer}
    
    Here is the retrieved context:
    {retrieved_context}
    
    Follow these steps:
    
    1. Extract all FACTUAL CLAIMS from the given answer.
    2. Put these claims in a list.
    3. Use the retrieved context to evaluate which claims are UNDISPUTED TRUTHS.
    4. Add the undisputed true facts to a list.
    
    Return the results in the following JSON format:
    {
        "factual_claims": [...],
        "undisputed_truths": [...]
    }
    
    Example:
    
    Question:
    What is the execution time for the task involving picking and placing red tomatoes with perception?
    
    Answer:
    The execution time for the task involving picking and placing red tomatoes with perception is not explicitly stated in the given context. However, it is mentioned that the execution time is the interval between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement.
    
    Retrieved Context:
    # Expected Output The video below shows the result of picking and placing red tomatoes with perception. 
    **You must submit video with full implementation** as shown. --> --> Note: **Execution time is the interval 
    between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence 
    mentioned in the Problem Statement. Refer Submission Instruction to know more.**
    
    Example Output JSON:
    
    {
        "factual_claims": [
            "The execution time for the task involving picking and placing red tomatoes with perception is not explicitly stated in the given context.",
            "However, it is mentioned that the execution time is the interval between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement."
        ],
        "undisputed_truths": [
            "The execution time is the interval between the first time the robots gets spawned and ends when the UR5 drops the last tomato as per the sequence mentioned in the Problem Statement."
        ]
    }
"""
