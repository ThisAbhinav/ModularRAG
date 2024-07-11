from abc import ABC, abstractmethod
import pandas as pd
from logger_config import logger
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from utils import getllamaIndexDocument, addMetadata
import os


class DataIngestion(ABC):
    @abstractmethod
    def load_data(self, datapath):
        pass


class MarkDownDataIngestion(DataIngestion):
    def load_data(self, datapath):
        """
        params:
        datapath: path to data directory
        returns: a dataframe with all text and metadata

        Loads data from each .md file and populates a df with text and metadata from each file
        """
        logger.info(f"Loading Markdown Data from {datapath}")
        try:
            text_loader_kwargs = {"autodetect_encoding": True}
            loader = DirectoryLoader(
                datapath,
                glob="**/*.md",
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs,
            )
            langchain_docs = loader.load()
            logger.info(f"Successfully loaded Markdown Data from {datapath}")
            texts = [doc.page_content for doc in langchain_docs]
            metadata = [addMetadata(doc.metadata, "markdown") for doc in langchain_docs]
            df = pd.DataFrame({"text": texts, "metadata": metadata})
            llamaindex_docs = getllamaIndexDocument(df)
            return df, langchain_docs, llamaindex_docs
        except Exception as e:
            logger.exception(
                f"Failed to load Markdown Data from {datapath} with error {e}"
            )
            raise


class PDFDataIngestion(DataIngestion):
    def load_data(self, datapath):
        """
        params:
        datapath: path to data directory containing PDF files
        returns: a dataframe with all text and metadata

        Loads data from all PDF files in the directory and populates a DataFrame with text and metadata from each page
        """
        logger.info(f"Loading PDF Data from directory: {datapath}")
        try:
            langchain_docs = []
            for filename in os.listdir(datapath):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(datapath, filename)
                    loader = PyPDFLoader(file_path)
                    langchain_docs.extend(loader.load())
            logger.info(f"Successfully loaded PDF Data from directory: {datapath}")
            texts = [doc.page_content for doc in langchain_docs]
            metadata = [addMetadata(doc.metadata, "pdf") for doc in langchain_docs]
            df = pd.DataFrame({"text": texts, "metadata": metadata})
            llamaindex_docs = getllamaIndexDocument(df)
            return df, langchain_docs, llamaindex_docs
        except Exception as e:
            logger.exception(
                f"Failed to load PDF Data from directory: {datapath} with error {e}"
            )
            raise
