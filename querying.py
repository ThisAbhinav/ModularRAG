from abc import ABC, abstractmethod
from logger_config import logger
import json
from utils import loadllm, messages_to_history_str
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from routing import Router

with open("config.json", "r") as file:
    config = json.load(file)


class Query(ABC):

    @abstractmethod
    def create_query(self, user_query):
        pass


class BaseQuery(Query):
    def create_query(self, user_query):
        return user_query


class MultiQuery(Query):
    def __init__(self, number_of_queries):
        self.number_of_queries = number_of_queries

    def create_query(self, user_query):
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
    def create_query(self, user_query):
        prompt = (
            f"For the given query, generate an answer from whatever you know, if you don't know, make up"
            f"an answer based on what you think might be the answer. Here is the query: {user_query}"
        )
        llm = loadllm(config["generator"]["name"])
        result = llm.complete(prompt)
        return (str(result) + " " + str(user_query))


class ChatQuery(Query):
    def create_query(self, user_query, chat_history):
        llm = loadllm("Ollama")

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
