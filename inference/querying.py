from abc import ABC, abstractmethod
from logger_config import logger
import json
from utils import loadllm, messages_to_history_str
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from routing import Router
from llama_index.core.tools import ToolMetadata

with open("config.json", "r") as file:
    config = json.load(file)


class Query(ABC):
    """Abstract base class for query objects."""

    @abstractmethod
    def create_query(self, user_query):
        """Create a query based on the user input.

        Args:
            user_query (str): The user's query.

        Returns:
            Union[str, List[str]]: The created query.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass


class BaseQuery(Query):
    """
    BaseQuery class represents a base query for querying data.

    Attributes:
        None

    Methods:
        create_query: Creates a query based on the user input.

    """
    def create_query(self, user_query):
        """
        Creates a query based on the user input.

        Args:
            user_query (str): The user input query.

        Returns:
            str: The created query.

        """
        return user_query


class MultiQuery(Query):
    """
    A class that represents a multi-query generator.

    Attributes:
        number_of_queries (int): The number of queries to generate.

    Methods:
        create_query(user_query: str) -> List[str]: Generates multiple queries based on a given user query.

    """
    def __init__(self, number_of_queries):
        self.number_of_queries = number_of_queries

    def create_query(self, user_query):
        """
        Generates multiple queries based on a given user query.

        Args:
            user_query (str): The user query to generate alternative queries for.

        Returns:
            List[str]: A list of generated queries.

        """
        prompt = (
            f"Given the below query, generate exactly 1 question that means the exact same but is worded "
            f"differently. Here is the query: {user_query}"
        )
        llm = loadllm(config["generator"]["name"])
        queries = []
        for _ in range(self.number_of_queries):
            result = llm.complete(prompt)
            queries.append(result)
        return queries


class HyDE(Query):
    """
    HyDE (Hybrid Data Exploration) class for creating queries and generating answers.

    Attributes:
        None

    Methods:
        create_query: Generates a query answer based on the given user query.

    """
    def create_query(self, user_query):
        """
        Generates a query answer based on the given user query.

        Args:
            user_query (str): The user query for which the answer needs to be generated.

        Returns:
            str: The generated answer for the user query.

        """
        prompt = (
            f"For the given query, generate an answer from whatever you know, if you don't know, make up"
            f"an answer based on what you think might be the answer. Here is the query: {user_query}"
        )
        llm = loadllm(config["generator"]["name"])
        result = llm.complete(prompt)
        return str(result)


class ChatQuery(Query):
    """
    Represents a query for a chat conversation.

    Attributes:
        user_query (str): The user's query.
        chat_history (List[str]): The chat conversation history.

    Methods:
        create_query: Creates a query based on the user's query and chat history.
    """
    def create_query(self, user_query, chat_history):
        """
        Creates a query based on the user's query and chat history.

        Args:
            user_query (str): The user's query.
            chat_history (List[str]): The chat conversation history.

        Returns:
            str: The generated query prompt.
        """
        llm = loadllm("Groq","mixtral-8x7b-32768")

        if len(chat_history) > 4:
            chat_history_summary = messages_to_history_str(chat_history[:-4])
            summarize_prompt = f"""Summarize the key points discussed in the conversation between the user and the chatbot, 
            including the main topics, any questions asked by the user and the chatbot's responses. 
            is the conversation: {chat_history_summary}"""
            chat_history_summary = llm.complete(summarize_prompt)

            chat_history_recent = messages_to_history_str(chat_history[-4:])
            chat_history_str = str(chat_history_summary) + " \n " + chat_history_recent
        else:
            chat_history_str = messages_to_history_str(chat_history)

        choices = [
            ToolMetadata(
                description="Chat conversation context is not related to the question and wouldn't directly help in answering this question",
                name="chat_history_unrelated",
            ),
            ToolMetadata(
                description="Chat conversation context is directly related to the question and would help in answering this question",
                name="chat_history_important",
            ),
        ]
        router = Router(llm)

        prompt = f"""Here is the CHAT CONVERSATION CONTEXT: 
            {chat_history_str}
            
            Here is the QUESTION : {user_query}
            """

        choice = router.route(prompt, choices)
        if choice == "chat_history_unrelated":
            prompt = user_query
        return str(prompt)
