from logger_config import logger
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.packs.raptor import RaptorPack
from abc import ABC, abstractmethod

DEFAULT_TOP_K = 5


class RetrieverModel(ABC):
    def __init__(self, top_k=DEFAULT_TOP_K):
        self.top_k = top_k
        self.retriever = None

    @abstractmethod
    def create_retriever(self, vector_index, embed_model):
        pass

    @abstractmethod
    def retrieve(self, query):
        pass


class BaseRetriever(RetrieverModel):
    def create_retriever(self, vector_index, embed_model, filters=None):
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
        try:
            logger.info(f"Retrieving for query: {query[:200]}...")
            results = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(results)} results")
            return results
        except Exception as e:
            logger.exception(f"Failed to retrieve with error: {e}")
            raise


class MultiQueryRetriever(RetrieverModel):
    def create_retriever(self, vector_index, embed_model, filters=None):
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


class RAPTOR(RetrieverModel):

    def create_retriever(self, vector_store, embed_model, llm, documents):
        self.retriever = RaptorPack(
            vector_store=vector_store,
            embed_model=embed_model,
            llm=llm,
            documents=documents,
        ).retriever
    
    def retrieve(self, query):
        return self.retriever.retrieve(query)
