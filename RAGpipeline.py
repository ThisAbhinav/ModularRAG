import pprint
from abc import ABC, abstractmethod
from logger_config import logger
import utils
import chunking
import retriever
import vectorstore
import dataingestion
from routing import Router
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from querying import ChatQuery
from retriever import RAPTOR


class RAGPipeline(ABC):
    def __init__(
        self,
        datatype,
        chunker,
        embed_model,
        vector_store,
        generator,
        retriever_type,
        datapath,
        query_method,
        reranker,
        use_cache=False,
        use_router=False,
        rerank=False,
        chat_mode=False,
        chat_history=[],
    ):
        self.datatype = datatype
        self.chunker = chunker
        self.vector_store = vector_store
        self.generator = generator
        self.embed_model = embed_model
        self.datapath = datapath
        self.retriever_type = retriever_type
        self.query_method = query_method
        self.use_router = use_router
        self.reranker = reranker
        self.rerank = rerank
        self.chat_history = chat_history
        self.chat_mode = chat_mode

        if use_cache:
            self.load_stored_data()
        else:
            if datatype == "pdf":
                self.data_ingestor = dataingestion.PDFDataIngestion()
            elif datatype == "markdown":
                self.data_ingestor = dataingestion.MarkDownDataIngestion()
            else:
                raise ValueError(f"Unsupported datatype: {datatype}")

            # Perform data ingestion and chunking during initialization
            self.load_and_process_data()

    def load_stored_data(self):
        logger.info(f"Using Cached data of collection default")
        self.index = self.vector_store.create_index_from_stored(
            embed_model=self.embed_model
        )
        if not self.use_router:
            if type(self.retriever_type) == type(RAPTOR()):
                llm = utils.loadllm("Ollama")
                vectorstore = self.vector_store.create_vector_store()
                self.retriever_type.create_retriever(
                    vector_store=vectorstore,
                    embed_model=self.embed_model,
                    llm=llm,
                    documents=[],
                )
            else:
                self.retriever_type.create_retriever(self.index, self.embed_model)

    def load_and_process_data(self):
        logger.info(f"Starting data ingestion from {self.datapath}")
        self.df, self.langchain_docs, self.llama_index_docs = (
            self.data_ingestor.load_data(self.datapath)
        )
        logger.info("Data ingestion completed")

        logger.info("Starting chunking process")
        self.chunks = self.chunker.chunk(self.llama_index_docs)

        # formatted_nodes = utils.nodeExtractor(
        #     retrieved_nodes, self.datatype
        # )  # This is not needed here

        llm = utils.loadllm("Ollama")

        if not self.use_router:
            if type(self.retriever_type) == type(RAPTOR()):
                llm = utils.loadllm("Ollama")
                vectorstore = self.vector_store.create_vector_store()
                self.retriever_type.create_retriever(
                    vector_store=vectorstore,
                    embed_model=self.embed_model,
                    llm=llm,
                    documents=self.llama_index_docs,
                )
            else:
                logger.info("Adding chunks to vector store")
                self.index = self.vector_store.create_index(
                    self.chunks, self.embed_model
                )
                logger.info("Adding chunks to vector store completed")
                logger.info("Chunking process completed")
                self.retriever_type.create_retriever(self.index, self.embed_model)

    def run(self, query):
        self.query = query
        logger.info(
            f"Running RAG Pipeline with data type {self.datatype} from {self.datapath}"
        )
        if self.chat_mode:
            logger.info("Chat mode enabled")
            query = ChatQuery().create_query(query, self.chat_history)

        transformed_query = self.query_method.create_query(query)
        # if self.use_router:
        #     llm = utils.loadllm("Groq")
        #     router = Router(llm)
        #     filter_word = router.route(query)
        #     if filter_word == "theme_missing":
        #         print("Theme is missing in the question")
        #         if self.chat_mode:
        #             self.chat_history.append(
        #                 ChatMessage(content=self.query, role=MessageRole.USER)
        #             )

        #         return (
        #             "I can't answer without more context related to the theme. Please provide theme. ",
        #             [],
        #             self.chat_history,
        #         )
        #     elif filter_word == "no_theme_required":
        #         filters = None
        #     else:
        #         filters = MetadataFilters(
        #             filters=[
        #                 ExactMatchFilter(key="theme", value=filter_word),
        #             ]
        #         )
        #     self.retriever_type.create_retriever(
        #         self.index, self.embed_model, filters=filters
        #     )
        retrieved_nodes = self.retriever_type.retrieve(transformed_query)
        if self.rerank:
            retrieved_nodes = self.reranker.run(query, retrieved_nodes)

        formatted_nodes = utils.nodeExtractor(retrieved_nodes, self.datatype)

        if "window" in retrieved_nodes[0].metadata:
            context_str = utils.getContextStringSentenceWindow(retrieved_nodes)
        else:
            context_str = utils.getContextString(retrieved_nodes)

        answer = self.generator.generate(context_str, query)
        logger.info(f"{self.datatype.capitalize()} RAG Pipeline completed")

        if self.chat_mode:
            self.chat_history.append(
                ChatMessage(content=self.query, role=MessageRole.USER)
            )
            self.chat_history.append(
                ChatMessage(content=answer, role=MessageRole.SYSTEM)
            )
        return answer, formatted_nodes, self.chat_history
