from llama_index.packs.raptor import RaptorPack
from llama_index.core.schema import TextNode
class RAPTOR():
    def __init__(self, llm, documents, embed_model, vector_store):
        self.pack = RaptorPack(documents=documents,llm=llm,embed_model=embed_model, vector_store=vector_store)
        return self.pack.retriever


class ContextualCompression():
    def __init__(self, llm):
        self.llm = llm

    def compressor(self, query_text: str, llama_index_doc):
        prompt = f"""Given the query: "{query_text}", compress the following document so that only information relevant to the query is retained.
        Document: {llama_index_doc.text}
        """
        response = self.llm.complete(prompt)
        return response
    
    def run(self, query_text: str, llama_index_docs):
        compressed_docs = TextNode()
        for doc in llama_index_docs:
            compressed_docs.text += self.compressor(query_text, doc)
        return compressed_docs