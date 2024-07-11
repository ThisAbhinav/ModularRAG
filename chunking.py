DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
# OLLAMA_BASE_URL = "http://10.129.152.197:11434"
OLLAMA_BASE_URL = "http://10.129.152.242:11434"
# BASE_OLLAMA_URL = "http://10.129.152.242:11434"

from abc import ABC, abstractmethod
from logger_config import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.node_parser import LangchainNodeParser, MarkdownNodeParser
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
)
from utils import loadEmbeddingModel, loadllm

# from langchain_community.chat_models import ChatOllama
import pprint
from tqdm import tqdm


class Chunking(ABC):
    @abstractmethod
    def chunk(self, data):
        pass


class RecursiveChunking(Chunking):

    def __init__(
        self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, llamaindex_docs):
        logger.info("Starting Recursive size chunking")
        splitter = LangchainNodeParser(
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        )
        try:
            chunks = splitter.get_nodes_from_documents(llamaindex_docs)
            logger.info("Recursive chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during recursive chunking")
            raise


class SemanticChunking(Chunking):
    def chunk(self, llamaindex_docs):
        logger.info("Starting semantic chunking")
        embed_model = loadEmbeddingModel("huggingface", "all-MiniLM-L6-v2")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )
        try:
            chunks = splitter.get_nodes_from_documents(
                llamaindex_docs, show_progress=True
            )
            logger.info("Semantic chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during semantic chunking")
            raise


class SentenceChunking(Chunking):
    def __init__(
        self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, llamaindex_docs):
        logger.info("Starting sentence chunking")
        splitter = SentenceSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        try:
            chunks = splitter.get_nodes_from_documents(llamaindex_docs)
            logger.info("Sentence chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during sentence chunking")
            raise


class HierarchicalChunking(Chunking):
    def chunk(self, llamaindex_docs):
        logger.info("Starting hierarchical chunking")
        splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[512, 256, 128], chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        try:
            chunks = splitter.get_nodes_from_documents(llamaindex_docs)
            logger.info("Hierarchical chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during hierarchical chunking")
            raise


class MarkdownChunking(Chunking):
    def chunk(self, llamaindex_docs):
        logger.info("Starting markdown chunking")
        splitter = MarkdownNodeParser()
        try:
            chunks = splitter.get_nodes_from_documents(llamaindex_docs)
            logger.info("Markdown chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during markdown chunking")
            raise


class SentenceWindowChunking(Chunking):
    def chunk(self, llamaindex_docs):
        logger.info("Starting sentence window chunking")
        splitter = SentenceWindowNodeParser.from_defaults(
            # how many sentences on either side to capture
            window_size=2,
            # the metadata key that holds the window of surrounding sentences
            window_metadata_key="window",
            # the metadata key that holds the original sentence
            original_text_metadata_key="original_sentence",
        )
        try:
            chunks = splitter.get_nodes_from_documents(llamaindex_docs)
            logger.info("Sentence window chunking completed")
            return chunks
        except Exception:
            logger.exception("Failed during sentence window chunking")
            raise


class AgenticChunking(Chunking):
    def __init__(
        self,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        llm=loadllm("Groq"),
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm

    def is_related(self, previous_chunk, current_chunk):
        prompt = """You are given two text chunks you have to determine if they are related or not. 
                    If they are related give output YES else NO. You must answer in one word , you will
                    get 1 point for each correct answer.
                    
                    chunk 1: {previous_chunk}
                    ////////////////////////////////////
                    chunk 2: {current_chunk}
                    """

        llm_response = self.llm.complete(
            prompt.format(previous_chunk=previous_chunk, current_chunk=current_chunk)
        )
        if "YES" in str(llm_response).upper():
            return True

    def chunk(self, llamaindex_docs):
        logger.info("Starting agentic chunking")
        splitter = SentenceSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        try:
            chunk_sentences = splitter.get_nodes_from_documents(llamaindex_docs)
            if not chunk_sentences:
                return []

            chunks = [chunk_sentences[0]]
            progress_bar = tqdm(chunk_sentences, desc="Chunking Progress", unit="chunk")

            for chunk in progress_bar:
                if len(chunks[-1].text) > 8198:
                    chunks.append(chunk)
                elif self.is_related(chunks[-1].text, chunk.text):
                    chunks[-1].text += chunk.text
                else:
                    chunks.append(chunk)

            logger.info("Agentic chunking completed")
            return chunks
        except Exception as e:
            logger.exception("Failed during agentic chunking")
            raise
