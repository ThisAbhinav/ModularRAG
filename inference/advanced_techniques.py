from llama_index.packs.raptor import RaptorPack
from llama_index.core.schema import TextNode
class RAPTOR():
    """
    RAPTOR class represents a retrieval-based question answering system.

    Args:
        llm (str): The language model used for generating embeddings.
        documents (list): List of documents used for retrieval.
        embed_model (str): The embedding model used for generating document embeddings.
        vector_store (str): The vector store used for storing document embeddings.

    Returns:
        retriever (object): An instance of the RaptorPack class.

    """
    def __init__(self, llm, documents, embed_model, vector_store):
        self.pack = RaptorPack(documents=documents,llm=llm,embed_model=embed_model, vector_store=vector_store)
        return self.pack.retriever


class ContextualCompression():
    """
    A class that performs contextual compression on documents based on a given query.

    Attributes:
        llm: The language model used for compression.

    Methods:
        compressor(query_text: str, llama_index_doc): Compresses a document based on the given query.
        run(query_text: str, llama_index_docs): Runs the contextual compression on a list of documents.

    """
    def __init__(self, llm):
        self.llm = llm

    def compressor(self, query_text: str, llama_index_doc):
        """
        Compresses a document based on the given query.

        Args:
            query_text (str): The query text.
            llama_index_doc: The document to be compressed.

        Returns:
            str: The compressed document.

        """
        prompt = f"""Given the query: "{query_text}", compress the following document so that only information relevant to the query is retained.
        Document: {llama_index_doc.text}
        """
        response = self.llm.complete(prompt)
        return response
    
    def run(self, query_text: str, llama_index_docs):
        """
        Runs the contextual compression on a list of documents.

        Args:
            query_text (str): The query text.
            llama_index_docs: The list of documents to be compressed.

        Returns:
            TextNode: The compressed documents.

        """
        compressed_docs = TextNode()
        for doc in llama_index_docs:
            compressed_docs.text += self.compressor(query_text, doc)
        return compressed_docs