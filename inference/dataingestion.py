from abc import ABC, abstractmethod
import pandas as pd
from logger_config import logger
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from utils import getllamaIndexDocument, addMetadata, append_neighbors_conditional,updateHash
import os
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

themedict = {
    "AB20.pdf": "Agri Bot",
    "AB23.pdf": "Astro Bot",
    "CL23.pdf": "Cosmo Logistic",
    "FR22.pdf": "Functional Weeder",
    "GG23.pdf": "Geo Guide",
    "HB23.pdf": "Hologlyph Bots",
    "LD23.pdf": "Luminosity Drone",
    "SB20.pdf": "Sahayak Bot",
    "SS21.pdf": "Vitaraṇa Drone",
    "VB20.pdf": "Vargi Bots",
    "VD20.pdf": "Vitarana Drone",
}

yeardict = {
    "AB20.pdf": 2020,
    "AB23.pdf": 2023,
    "CL23.pdf": 2023,
    "FR22.pdf": 2022,
    "GG23.pdf": 2023,
    "HB23.pdf": 2023,
    "LD23.pdf": 2023,
    "SB20.pdf": 2020,
    "SS21.pdf": 2021,
    "VB20.pdf": 2020,
    "VD20.pdf": 2020,
}


class DataIngestion(ABC):
    """
    Abstract base class for data ingestion.

    This class provides an interface for loading data from a given datapath.

    Attributes:
        None

    Methods:
        load_data: Abstract method to load data from a given datapath.

    """
    @abstractmethod
    def load_data(self, datapath):
        pass


class MarkDownDataIngestion(DataIngestion):
    """
    A class for ingesting Markdown data.

    This class extends the DataIngestion class and provides methods to load Markdown data
    from a specified directory and populate a dataframe with text and metadata from each file.
    """
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
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        logger.info(f"Loading PDF Data from directory: {datapath}")
        try:
            langchain_docs = []
            for filename in os.listdir(datapath):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(datapath, filename)
                    pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
                    documents = pdf_loader.load_data(file_path)
                    # loader = PyMuPDFLoader(file_path)
                    # langchain_docs.extend(loader.load())
                    pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
                    documents = pdf_loader.load_data(file_path)
                    documents = [doc.to_langchain_format() for doc in documents]
                    print(documents)
                    langchain_docs.extend(documents)
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


class PDFMarkdownDataIngestion(DataIngestion):
    def load_data(self, datapath):
        """
        params:
        datapath: path to data directory containing PDF files
        returns: a dataframe with all text and metadata

        Loads data from all PDF files in the directory and populates a DataFrame with text and metadata from each page
        """
        logger.info(f"Loading PDF Data from directory: {datapath}")
        try:
            headers_to_split_on = [
                ("#", "heading"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            themesummaries = []
            taskschunks = []
            for filename in os.listdir(datapath):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(datapath, filename)
                    logger.info(f"Loading {filename}")
                    md_text = pymupdf4llm.to_markdown(file_path)

                md_header_splits = markdown_splitter.split_text(md_text)
                start = 0
                themesummary = []
                for i in range(len(md_header_splits)):
                    if (
                        "welcome to" in md_header_splits[i].metadata["heading"].lower()
                        or "introduction"
                        in md_header_splits[i].metadata["heading"].lower()
                        or "theme description"
                        in md_header_splits[i].metadata["heading"].lower()
                    ):
                        themesummary += "\n" + md_header_splits[i].page_content

                    if (
                        "task" in md_header_splits[i].metadata["heading"].lower()
                        or i == len(md_header_splits) - 1
                    ):
                        temp = f"\n Heading: {md_header_splits[start].metadata['heading']} \n "
                        for j in range(start, i):
                            temp += f" \n {md_header_splits[j].metadata['heading']} - Content: {md_header_splits[j].page_content}"
                        taskschunks.append(
                            Document(
                                text=temp,
                                metadata={
                                    "task": md_header_splits[start].metadata["heading"],
                                    "theme": themedict[filename],
                                    "year": yeardict[filename],
                                },
                            )
                        )
                        start = i
                themesummaries.append(themesummary)
            return themesummaries, taskschunks
        except Exception as e:
            logger.exception(
                f"Failed to load PDF Data from directory: {datapath} with error {e}"
            )
            raise


class LLMSherpaIngestion(DataIngestion):
    def load_data(self, datapath):
        docs = []
        try:
            for i, filename in enumerate(os.listdir(datapath)):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(datapath, filename)
                    while True:
                        try:
                            print("Trying: ", filename)
                            loader = LLMSherpaFileLoader(
                                file_path=file_path,
                                new_indent_parser=True,
                                apply_ocr=False,
                                strategy="chunks",
                                llmsherpa_api_url=llmsherpa_api_url,
                            )
                            documents = loader.load()
                            break
                        except Exception as e:
                            print(f"LLM SHERPA FAILED:  {e} \n TRYING AGAIN !!!")
                            
                            
                    texts = [doc.page_content for doc in documents]
                    metadata = [addMetadata(doc.metadata, "pdf") for doc in documents]
                    df = pd.DataFrame({"text": texts, "metadata": metadata})
                    llamaindex_docs = getllamaIndexDocument(df)
                    llamaindex_docs = append_neighbors_conditional(llamaindex_docs)
                    docs.extend(llamaindex_docs)
            return df, [] , docs
        except Exception as e:
            print(f"Failed to load PDF Data with error {e}")
                  
    def load_data_single_file(self, datapath):
        try:
            loader = LLMSherpaFileLoader(
                
                        file_path=datapath,
                        new_indent_parser=True,
                        apply_ocr=False,
                        strategy="chunks",
                        llmsherpa_api_url=llmsherpa_api_url,
                    )
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
            metadata = [doc.metadata for doc in documents]
            df = pd.DataFrame({"text": texts, "metadata": metadata})
            llamaindex_docs = getllamaIndexDocument(df)
            llamaindex_docs = append_neighbors_conditional(llamaindex_docs)
            return llamaindex_docs
        except Exception as e:
            print(f"Failed to load PDF Data with error {e}")
            exit()