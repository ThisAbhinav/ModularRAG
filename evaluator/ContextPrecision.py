from abc import ABC, abstractmethod, abstractproperty
from utils import loadllm
from llama_index.core.llms import ChatMessage
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

llm = loadllm("Groq", temperature=0)
def context_precision(query, ground_truth, retrieved_contexts, retries=3):


    system_prompt = """
    You are an expert text relevancy evaluator. Given a question, the ground truth, and a list of passages 
    (retrieved contexts) along with their indices,analyze each sentence by comparing it with the ground truth,
    and evaluate its relevance to the question
    """

    prompt = """
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.

    Given the question:
    {query}
    
    Here is the ground truth:
    {ground_truth}
    
    Here are the retrieved contexts with their respective indices:
    {retrieved_contexts}
    
    Follow these steps:
    
    1. Evaluate each context in the retrieved contexts list for relevance to the question based on the ground truth.
    2. For each relevant context, note its index which has been given along with it.
    3. Collect all indices of relevant contexts into a list.
    
    Return the results in the following JSON format:
    {{
        "relevant_indices": [...]
    }}
    
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.
    
    Here is an example:
    
    Question:
    What type of gripper is used on the Agribot and what is its stroke?
    
    Ground Truth:
    The Agribot uses a two-finger adaptive gripper with a stroke of 8.5 cm, allowing it to hold objects of that size between its fingers.
    
    Retrieved Contexts:
    
    0. #####"a radius of 850 mm, it is the perfect cobot for performing light tasks such as packing, assembly, or testing.\nRobotic gripper:\nThe robotic gripper used on the Agribot is a two-ﬁnger adaptive type. It\nhas a stroke of 8.5cm i.e. it can hold an object of size 8.5 cm between its\ntwo ﬁngers. The gripper structure has a cup-shaped attachment at its\nﬁngertip to pluck the tomatoes."
    1. #####"Figure 1: Agribot Manipulation\nRobotic Gripper:\nOn the tool tip of UR5 a 2 ﬁnger gripper(2f Robotiq gripper) is\nconnected.\nFor conﬁguring the gripper in MoveIt set up assistant use only\ngripper_finger1_joint (for gripper planning group) to control the\nmotion of gripper.\nOther joints of the gripper: gripper_finger1_inner_knuckle_joint,\ngripper_finger1_finger_tip_joint,\ngripper_finger2_inner_knuckle_joint,"
    2. #####"UR5 robotic arm:\nThe Universal Robots UR5 is a highly ﬂexible robotic arm that enables safe automation of repetitive, risky tasks. With a carrying capacity of 5 KG and a\nradius of 850 mm, it is the perfect cobot for performing light tasks such as\npacking, assembly or testing.\nRobotic gripper:\nThe robotic gripper used on the Sahayak Bot is a 2F type. It has a stroke of\n8.5cm i.e. it can hold an object of size 8.5 cm between its two ﬁngers."
    3. #####"Trump is the president of USA"
    4. #####"The Agribot has a dimension of 100cm by 100cm and is used for a variety of agricultural tasks."
    
    Example JSON:
    {{
        "relevant_indices": [0, 1]
    }}
    **IMPORTANT**: Strictly return only the JSON formatted results without any surrounding text.


    """

    for attempt in range(retries):
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt.format(query=query, retrieved_contexts=retrieved_contexts, ground_truth=ground_truth)),
        ])
        print(prompt.format(query=query, retrieved_contexts=retrieved_contexts, ground_truth=ground_truth))

        try:
            result = json.loads(str(response.message.content))
            print(response.message.content)
            break
        except json.JSONDecodeError:
            result = extract_json_manually(str(response.message.content))
            print(response.message.content)
            if result:
                break
    else:
        print(f"Failed to parse response as JSON after {retries} attempts.")
        print("Response:", response.message.content)
        return


    relevant_indices = result["relevant_indices"]
    if len(relevant_indices) == 0:
        cumulative_precision = 0
    else:
        relevant_nodes = 0
        sum = 0
        for i in range(len(eval(retrieved_contexts))):
            k=i+1
            relevancy = 0
            if i in relevant_indices:
                relevant_nodes += 1
                relevancy = 1
            sum += (relevant_nodes/k)*relevancy
        cumulative_precision = (sum/len(relevant_indices))

    return cumulative_precision


def extract_json_manually(text):
    try:
        match = re.search(r'\{.*?}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None


def cosine_sim(sentence1, sentence2):
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]


df = pd.read_csv("/home/deepakachu/Desktop/RAG/26_Assessment_RAG/results/pipeline_2024-06-12_13-08_pdffffadfasfa.csv")
eg = df.iloc[1]

query = eg["question"]
retrieved_contexts = eg["contexts"]
ground_truth = eg["ground_truth"]

print(query)
print(len(eval(retrieved_contexts)))
print(len(retrieved_contexts))
print(ground_truth)

combined_text = ""
for i, passage in enumerate(eval(retrieved_contexts)):
    combined_text += f"{i}. ######{passage}\n\n"
print(context_precision(query, ground_truth, combined_text))