from logger_config import logger
from llama_index.core.retrievers import VectorIndexRetriever
from abc import ABC, abstractmethod

DEFAULT_TOP_K = 5


class RetrieverModel(ABC):
    """
    Abstract base class for retriever models.
    """
    def __init__(self, top_k=DEFAULT_TOP_K):
        """
        Initializes a RetrieverModel object.

        Args:
            top_k (int): The number of top results to retrieve. Defaults to DEFAULT_TOP_K.
        """
        self.top_k = top_k
        self.retriever = None

    @abstractmethod
    def create_retriever(self, vector_index, embed_model):
        """
        Abstract method to create a retriever.

        Args:
            vector_index: The vector index used for retrieval.
            embed_model: The embedding model used for retrieval.
        """
        pass

    @abstractmethod
    def retrieve(self, query):
        """
        Abstract method to retrieve results based on a query.

        Args:
            query (str): The query used for retrieval.

        Returns:
            List[Union[dict, str]]: A list of retrieved results.
        """
        pass


class BaseRetriever(RetrieverModel):
    """
    Base class for retrievers.

    This class provides the basic functionality for creating and retrieving results using a retriever model.

    Args:
        vector_index (VectorIndex): The vector index used for retrieval.
        embed_model (EmbedModel): The embedding model used for retrieval.
        filters (Optional[List[Filter]]): A list of filters to apply during retrieval.
    """
    def create_retriever(self, vector_index, embed_model, filters=None):
        """
        Create a retriever with the specified vector index, embedding model, and filters.

        Args:
            vector_index (VectorIndex): The vector index used for retrieval.
            embed_model (EmbedModel): The embedding model used for retrieval.
            filters (Optional[List[dict]]): A list of filters to apply during retrieval.
        """
        logger.info(f"Creating VectorIndexRetriever with top_k={self.top_k}")
        try:
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=self.top_k,
                embed_model=embed_model,
                filters=filters,
                verbose=True,
            )
            logger.info("VectorIndexRetriever created successfully")
            self.retriever = retriever
        except Exception as e:
            logger.exception(f"Failed to create VectorIndexRetriever with error: {e}")
            raise

    def retrieve(self, query):
        """
        Retrieve results for the given query.

        Args:
            query (str): The query to retrieve results for.

        Returns:
            List[Union[dict, str]]: The retrieved results.
        """
        try:
            logger.info(f"Retrieving for query: {query[:200]}...")
            results = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.exception(f"Failed to retrieve with error: {e}")
            raise


class MultiQueryRetriever(RetrieverModel):
    """
    A class for handling multiple queries and retrieving results.

    Args:
        vector_index (VectorIndex): The vector index used for retrieval.
        embed_model (EmbedModel): The embedding model used for retrieval.
        filters (Optional[List[Filter]]): A list of filters to apply during retrieval.
    """
    def create_retriever(self, vector_index, embed_model, filters=None):
        """
        Create a MultiIndexRetriever with the specified parameters.

        Args:
            vector_index: The vector index used for retrieval.
            embed_model: The embedding model used for retrieval.
            filters: Optional filters to apply during retrieval.
        """
        logger.info(f"Creating MultiIndexRetriever with top_k={self.top_k}")
        try:
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=self.top_k,
                embed_model=embed_model,
                filters=filters,
            )
            logger.info("VectorIndexRetriever created successfully")
            self.retriever = retriever
        except Exception as e:
            logger.exception(f"Failed to create VectorIndexRetriever with error: {e}")
            raise

    def retrieve(self, queries):
        """
        Retrieves results for the given queries.

        Args:
            queries (List[str]): A list of queries to retrieve results for.

        Returns:
            List[Union[dict, str]]: A list of retrieved results.

        Raises:
            TypeError: If queries is not a list.
        """
        logger.info(f"Retrieving for queries: {queries}")
        if not isinstance(queries, list):
            logger.error("MultiQueryRetriever requires a list of queries")
        try:
            results = []
            for query in queries:
                logger.info(f"Retrieving for query: {query[:200]}...")
                results.extend(self.retriever.retrieve(query))
                logger.info(f"Retrieval done for query: {query[:200]}...")
            return results
        except Exception as e:
            logger.exception(f"Failed to retrieve with exception {e}")






