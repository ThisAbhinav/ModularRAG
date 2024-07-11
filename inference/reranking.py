from flashrank import Ranker, RerankRequest
from llama_index.core.schema import TextNode,NodeWithScore
from abc import ABC, abstractmethod
from utils import loadllm
class RerankerModel(ABC):
    """
    Abstract base class for a reranker model.

    Args:
        llm (object): The language model used for reranking.
        top_k (int): The number of top documents to consider during reranking.
        model_name (str, optional): The name of the model. Defaults to None.
    """
    def __init__(self, llm, top_k, model_name=None):
        self.model_name = model_name
        self.top_k = top_k
        self.llm = llm

    @abstractmethod
    def run(self, query_text: str, llama_index_docs):
        """
        Abstract method to run the reranker model.

        Args:
            query_text (str): The query text to rerank the documents for.
            llama_index_docs (list): The list of documents to rerank.

        Returns:
            list: The reranked documents.
        """
        pass


class CrossEncoderReranker(RerankerModel):
    """
    A class representing a cross-encoder reranker model.

    This class inherits from the `RerankerModel` base class and provides methods to rerank a list of documents based on a given query.

    Attributes:
        model_name (str): The name of the model.
    """
    def reranker(self):
        return Ranker(model_name=self.model_name, cache_dir="./cache")

    def run(self, query_text: str, llama_index_docs):
        """
        Reranks a list of documents based on a given query.

        Args:
            query_text (str): The query text.
            llama_index_docs (list): A list of documents to be reranked.

        Returns:
            list: The reranked documents.
        """
        formatted_data = [{"text": doc.text} for doc in llama_index_docs]
        rerankrequest = RerankRequest(query=query_text, passages=formatted_data)
        ranker = self.reranker()
        reranked = ranker.rerank(rerankrequest)
        
        reranked_nodes = []
        for result in reranked:
            for doc in llama_index_docs:
                if doc.text == result["text"]:
                    doc.score = result["score"] 
                    reranked_nodes.append(doc)
        return reranked_nodes[:min(self.top_k, len(reranked_nodes))]
                    

class PairWiseReRanker(RerankerModel):
    """
    A class that performs pairwise reranking of documents based on relevance to a given query.
    """
    
    def reranker(self, query_text: str, doc_a, doc_b):
        """
        Reranks two documents based on their relevance to a given query.

        Args:
            query_text (str): The query text.
            doc_a: The first document to compare.
            doc_b: The second document to compare.

        Returns:
            str: The character 'B' if doc_b is more relevant, otherwise None.
        """
        prompt = f"""Given the query: "{query_text}", which of the following documents is most relevant? Give one character answer: Either A or B.
        A: {doc_a.text}
        ////////////////////////////////////////////////////////////////////////////////
        B: {doc_b.text}
        """
        response = self.llm.complete(prompt)
        if 'B' in response:
            return 'B'
    def bubble_sort_docs(self, query_text: str, llama_index_docs):
        """
        Sorts a list of documents based on their relevance to a given query using bubble sort algorithm.

        Args:
            query_text (str): The query text.
            llama_index_docs: The list of documents to sort.

        Returns:
            list: The sorted list of documents.
        """
        n = len(llama_index_docs)
        for i in range(n):
            for j in range(0, n-i-1):
                if self.reranker(query_text, llama_index_docs[j], llama_index_docs[j+1]) == 'B':
                    llama_index_docs[j], llama_index_docs[j+1] = llama_index_docs[j+1], llama_index_docs[j]
        return llama_index_docs

    def run(self, query_text: str, llama_index_docs):
        """
        Runs the pairwise reranking algorithm on a list of documents.

        Args:
            query_text (str): The query text.
            llama_index_docs: The list of documents to rerank.

        Returns:
            list: The top-k reranked documents.
        """
        reranked_nodes = self.bubble_sort_docs(query_text, llama_index_docs)
        
        return reranked_nodes[:min(self.top_k, len(reranked_nodes))]
    
class ListWiseReranker(RerankerModel):
    def reranker(self, query_text: str, llama_index_docs):
        """
        Reranks the documents based on the given query text and returns them in decreasing order of relevance.

        Args:
            query_text (str): The query text to be used for reranking.
            llama_index_docs (list): A list of documents to be reranked.

        Returns:
            list: The reranked documents in decreasing order of relevance.
        """
        prompt = f"""Given the query: "{query_text}", Give the documents in decreasing order of relevance. You must give list of identifiers like A, B, C, D, E. 
        A: {llama_index_docs[0].text}
        ////////////////////////////////////////////////////////////////////////////////
        B: {llama_index_docs[1].text}
        ////////////////////////////////////////////////////////////////////////////////
        C: {llama_index_docs[2].text}
        ////////////////////////////////////////////////////////////////////////////////
        D: {llama_index_docs[3].text}
        ////////////////////////////////////////////////////////////////////////////////
        E: {llama_index_docs[4].text}
        """
        response = self.llm.complete(prompt)
        sorted_indices = []
        reranked_docs = []
        if 'A' in response:
            sorted_indices.append(response.index('A'))
        if 'B' in response:
            sorted_indices.append(response.index('B'))
        if 'C' in response:
            sorted_indices.append(response.index('C'))
        if 'D' in response:
            sorted_indices.append(response.index('D'))
        if 'E' in response:
            sorted_indices.append(response.index('E'))
        sorted_indices.sort()
        for index in sorted_indices:
            reranked_docs.append(llama_index_docs[index])
        return reranked_docs
            
    def run(self, query_text: str, llama_index_docs):
        """
        Runs the reranking process on the given query text and llama index documents.

        Args:
            query_text (str): The query text to be reranked.
            llama_index_docs: The llama index documents to be used for reranking.

        Returns:
            list: The reranked nodes, limited to the top k nodes.
        """
        reranked_nodes = self.reranker(query_text, llama_index_docs)
        return reranked_nodes[:min(self.top_k, len(reranked_nodes))]

