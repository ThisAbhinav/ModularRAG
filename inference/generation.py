from logger_config import logger
from abc import ABC, abstractmethod
from utils import loadllm

# DEFAULT_PROMPT = """
# You are an expert assistant with access to a vast amount of information. Based on the following context, answer the query accurately and comprehensively.\n\n

#     CONTEXT: \n{context}\n\n
#     \n
#     QUESTION: \n{query}\n\n

# Please provide a detailed and informative response based on the given context.
# Remember, don't blindly repeat the contexts verbatim and don't tell the user how you used the citations or context- just respond with the answer.
# It is very important for my career that you follow these instructions. Also don't mention the context in your response.

# """
# DEFAULT_PROMPT = """ Given a user question and some context, please write a clean, concise and accurate answer to the question based on the context.

# You will be given a set of related contexts to the question. Please use the context when crafting your answer.

# Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

# eg. if the question is ambiguous and doesn't specify the theme  and just mentions a task then you can say "Not enough information".

# Here are the set of contexts: {context}

# Remember, don't blindly repeat the contexts verbatim and don't tell the user how you used the citations or context- just respond with the answer. It is very important for my career that you follow these instructions. Here is the user question: {query} """


# print(DEFAULT_LLM)

DEFAULT_PROMPT = """
You are an expert AI e-Yantra assistant with access to a vast amount of information. e-Yantra is a robotics outreach program hosted at IIT Bombay and funded by the Ministry of Education. 
The goal is to harness the talent of young engineers to solve problems using technology across a variety of domains.
Based on the following CHAT HISTORY, CONTEXT, answer the QUESTION accurately and comprehensively. If chat history is irrelevant to the query then ignore it. \n\n 
    
    CHAT HISTORY: \n{history}\n\n
    \n
    CONTEXT: \n{context}\n\n
    \n
    QUESTION: \n{query}\n\n
    
Please provide a detailed and informative response based on the given context. 
Remember, don't blindly repeat the contexts verbatim and don't tell the user how you used the citations or context- just respond with the answer. 
It is very important for my career that you follow these instructions. Also don't mention the context in your response.

"""


class Generator(ABC):
    """
    Abstract base class for generators.

    Attributes:
        llm (LLM): The language model used by the generator.
        prompt (str): The default prompt for generating text.

    Methods:
        generate(context, query): Abstract method for generating text based on the given context and query.
    """
    def __init__(self, llm, prompt=DEFAULT_PROMPT):
        self.llm = llm
        self.prompt = prompt

    @abstractmethod
    def generate(self, context, query, history=""):
        pass


class SimpleGenerator(Generator):
    """
    A simple generator class that generates text based on a given context and query.

    Args:
        Generator (class): The base generator class.

    Methods:
        generate(context, query): Generates text based on the provided context and query.

    """
    def generate(self, context, query,history=""):
        """
        Generates text based on the provided context and query.

        Args:
            context (str): The context for text generation.
            query (str): The query for text generation.

        Returns:
            str: The generated text.

        """
        return str(self.llm.complete(self.prompt.format(context=context, query=query,history=history)))
