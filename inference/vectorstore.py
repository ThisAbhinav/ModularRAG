import shutil
from abc import ABC, abstractmethod
from utils import loadEmbeddingModel,updateHash
import chromadb, os
from logger_config import logger
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

DATABASE_PATH = "data/vector"


class VectorStore(ABC):
    def __init__(self, database_path=DATABASE_PATH, name="default"):
        self.database_path = database_path
        self.name = name

    def create_index_from_stored(self, embed_model):
        """
        Creates an index from the stored vector store.

        Args:
            embed_model: The embedding model used to create the index.

        Returns:
            The created vector store index.

        Raises:
            None.
        """
        logger.info("Index already exists")
        client = chromadb.PersistentClient(path=os.path.join(self.database_path, self.name))
        collection = client.get_collection(self.name)
        vectorstore = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vectorstore, embed_model=embed_model, show_progress=True
        )
        logger.info("Vector store index created")
        return index

    def create_index(self, llama_index_nodes, embed_model,update=False):
        """
        Creates an index for the vector store.

        Args:
            llama_index_nodes (list): A list of LlamaIndexNode objects representing the nodes in the index.
            embed_model: The embedding model used for vector representation.

        Returns:
            VectorStoreIndex: The created vector store index.

        Raises:
            OSError: If the index directory cannot be deleted.
        """
        logger.info("Starting index creation process")
        if (not update) and os.path.exists(os.path.join(self.database_path, self.name)):
            try:
                shutil.rmtree(os.path.join(self.database_path, self.name))
                logger.info("Index already existed, so deleting before creating again")
            except Exception as e:
                logger.info(f"Index existed but could not delete, this WILL cause retrieval to fail,"
                            f"please resolve manually: {e}")

        chroma_client = chromadb.PersistentClient(path=os.path.join(self.database_path, self.name))
        logger.info("Created Chroma persistent client")

        collection = chroma_client.get_or_create_collection(self.name)
        logger.info(f"Created or retrieved collection {self.name}")

        vectorstore = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(vector_store=vectorstore)
        logger.info("Storage context created")
        print(llama_index_nodes[0])
        llama_index_nodes = updateHash(llama_index_nodes)
        index = VectorStoreIndex(
            nodes=llama_index_nodes,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.info("Vector store index created")

        return index
