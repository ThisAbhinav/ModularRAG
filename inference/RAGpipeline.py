import pprint
from abc import ABC, abstractmethod
from logger_config import logger
import utils
import chunking
import retriever
from vectorstore import VectorStore
import dataingestion
from routing import Router
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from querying import ChatQuery

class RAGPipeline(ABC):
    def __init__(
        self,
        datatype,
        ingestor,
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
    ) -> None:
        """
        Initialize the RAGPipeline object.

        :param datatype: The type of data to be processed. Supported values are "pdf" and "markdown".
        :param chunker: The chunker object used for data chunking.
        :param embed_model: The embedding model used for text embedding.
        :param vector_store: The vector store used for storing and retrieving embeddings.
        :param generator: The generator model used for text generation.
        :param retriever_type: The type of retriever used for query retrieval.
        :param datapath: The path to the data to be processed.
        :param query_method: The method used for querying the retriever.
        :param reranker: The reranker model used for reranking the retrieved results.
        :param use_cache: Whether to use cached data. Defaults to False.
        :param use_router: Whether to use a router for query routing. Defaults to False.
        :param rerank: Whether to perform reranking. Defaults to False.
        :param chat_mode: Whether to enable chat mode. Defaults to False.
        :param chat_history: The chat history. Defaults to an empty list.
        """
        self.datatype = datatype
        self.ingestor = ingestor
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
        self.added_file = ""
        if use_cache:
            self.load_stored_data()
        else:
            self.load_and_process_data()

    def load_stored_data(self) -> None:
        """
        Loads the stored data and creates an index for retrieval.

        This method loads the cached data of the collection and creates an index
        for retrieval using the specified embedding model. If the `use_router`
        flag is set to False, it also creates a retriever using the index and
        embedding model.
        """
        logger.info(f"Using Cached data of collection")
        self.index = self.vector_store.create_index_from_stored(
            embed_model=self.embed_model
        )
        if self.added_file or not self.use_router:
            self.retriever_type.create_retriever(self.index, self.embed_model)

    def load_and_process_data(self) -> None:
        """
        Loads and processes data from the specified datapath.

        This method performs the following steps:
        
        1. Logs the start of data ingestion.
        2. Calls the `load_data` method of the `data_ingestor` object to load the data from the datapath.
        3. Assigns the loaded data to the `df`, `langchain_docs`, and `llama_index_docs` attributes.
        4. Logs the completion of data ingestion.
        5. Logs the start of the chunking process.
        6. Calls the `chunk` method of the `chunker` object to chunk the `llama_index_docs`.
        7. Assigns the chunks to the `chunks` attribute.
        8. Logs the completion of the chunking process.
        9. Logs the start of adding chunks to the vector store.
        10. Calls the `create_index` method of the `vector_store` object to add the chunks to the vector store.
        11. Logs the completion of adding chunks to the vector store.
        12. If `use_router` is False, calls the `create_retriever` method of the `retriever_type` object to create a retriever using the index and embed_model.
        """
        logger.info(f"Starting data ingestion from {self.datapath}")
        self.df, self.langchain_docs, self.llama_index_docs = (
            self.ingestor.load_data(self.datapath)
        )
        logger.info("Data ingestion completed")
        if isinstance(self.ingestor, dataingestion.LLMSherpaIngestion):
            logger.info("LLM Sherpa data ingestion detected. Skipping chunking process")
            self.chunks = self.llama_index_docs
        else:
            logger.info("Starting chunking process")
            self.chunks = self.chunker.chunk(self.llama_index_docs)

        # formatted_nodes = utils.nodeExtractor(retrieved_nodes, self.datatype)
        logger.info("Chunking process completed")

        logger.info("Adding chunks to vector store")
        self.index = self.vector_store.create_index(self.chunks, self.embed_model)


        logger.info("Adding chunks to vector store completed")
        if self.added_file or not self.use_router:
            self.retriever_type.create_retriever(self.index, self.embed_model)
        logger.info("added chunks now restart the pipeline after changing to cache = true in the pipeline config file")
        exit()

    def run(self, query) -> tuple:
        """
        Runs the RAG Pipeline with the given query.

        :param query: The query to be processed.
        :returns: A tuple containing the answer, formatted nodes, and chat history.
        """
        self.query = query
        logger.info(
            f"Running RAG Pipeline with data type {self.datatype} from {self.datapath}"
        )
        if self.chat_mode:
            logger.info("Chat mode enabled")
            query = ChatQuery().create_query(query, self.chat_history)

        transformed_query = self.query_method.create_query(query)

        if self.use_router and not self.added_file:
            llm = utils.loadllm("Groq","llama3-70b-8192")
            router = Router(llm)
            filter_word = router.route(query)
            logger.warning(f"Filter word: {filter_word}")
            # if filter_word == "theme_missing":
            #     print("Theme is missing in the question")
            #     if self.chat_mode:
            #         self.chat_history.append(
            #             ChatMessage(content=self.query, role=MessageRole.USER)
            #         )

            #     return (
            #         "I can't answer without more context related to the theme. Please provide theme. ",
            #         [],
            #         self.chat_history,
            #     )
            if filter_word == "no_theme_required" or filter_word == "theme_missing":
                filters = None
            else:
                filters = MetadataFilters(
                    filters=[
                        ExactMatchFilter(key="theme", value=filter_word),
                    ]
                )
            self.retriever_type.create_retriever(
                self.index, self.embed_model, filters=filters
            )
        else:
            if self.added_file:
                filter_word = self.added_file
                logger.warning(f"Filter word: {filter_word}")
                filters = MetadataFilters(
                    filters=[
                        ExactMatchFilter(key="source", value=filter_word),
                    ]
                )
                self.retriever_type.create_retriever(
                    self.index, self.embed_model, filters=filters
                )
            else:
                self.retriever_type.create_retriever(self.index, self.embed_model)
        retrieved_nodes = self.retriever_type.retrieve(transformed_query)
        # print(retrieved_nodes)

        if self.rerank:
            retrieved_nodes = self.reranker.run(query, retrieved_nodes)
        # print(retrieved_nodes)

        formatted_nodes = utils.nodeExtractor(retrieved_nodes, self.datatype)

        if len(retrieved_nodes) and  "window" in retrieved_nodes[0].metadata:
            context_str = utils.getContextStringSentenceWindow(retrieved_nodes)
        else:
            context_str = utils.getContextString(retrieved_nodes)
        if self.chat_mode:
            answer = self.generator.generate(context_str, query, self.chat_history)
        else: 
            answer = self.generator.generate(context_str, query,"")
            
        logger.info(f"{self.datatype.capitalize()} RAG Pipeline completed")

        if self.chat_mode:
            self.chat_history.append(
                ChatMessage(content=self.query, role=MessageRole.USER)
            )
            self.chat_history.append(
                ChatMessage(content=answer, role=MessageRole.SYSTEM)
            )
        return answer, formatted_nodes, self.chat_history
 
    def add_file(self,path):
        self.added_file = path
        print("adding file to the pipeline! ")
        docs=dataingestion.LLMSherpaIngestion().load_data_single_file(path)
        print(docs)

        self.index = self.vector_store.create_index(docs,self.embed_model,update=True)
        print("file added to the pipeline! ")