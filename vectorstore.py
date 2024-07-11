import shutil
from abc import ABC, abstractmethod
from utils import loadEmbeddingModel
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
    def create_vector_store(self):
        client = chromadb.PersistentClient(path=os.path.join(self.database_path, self.name))
        collection = client.get_or_create_collection(self.name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return vector_store
    def create_index_from_stored(self, embed_model):
        logger.info("Index already exists")
        client = chromadb.PersistentClient(path=os.path.join(self.database_path, self.name))
        collection = client.get_collection(self.name)
        vectorstore = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vectorstore, embed_model=embed_model, show_progress=True
        )
        logger.info("Vector store index created")
        return index

    def create_index(self, llama_index_nodes, embed_model):
        logger.info("Starting index creation process")
        if os.path.exists(os.path.join(self.database_path, self.name)):
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

        index = VectorStoreIndex(
            nodes=llama_index_nodes,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.info("Vector store index created")

        return index
