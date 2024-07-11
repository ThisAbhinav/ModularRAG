from flashrank import Ranker, RerankRequest
from llama_index.core.schema import TextNode
from abc import ABC, abstractmethod
from utils import loadllm
class RerankerModel(ABC):
    def __init__(self, llm, top_k, model_name=None):
        self.model_name = model_name
        self.top_k = top_k
        self.llm = llm

    @abstractmethod
    def run(self, query_text: str, llama_index_docs):
        pass
    

class CrossEncoderReranker(RerankerModel):
    def reranker(self):
        return Ranker(model_name=self.model_name, cache_dir="./cache")
    
    def run(self, query_text: str, llama_index_docs):
        formatted_data = [{"text": doc.text} for doc in llama_index_docs]
        rerankrequest =  RerankRequest(query=query_text, passages=formatted_data)
        ranker = self.reranker()
        reranked = ranker.rerank(rerankrequest)
        print(reranked)
        reranked_nodes = []
        for result in reranked:
            for doc  in llama_index_docs:
                if doc.text == result["text"]:
                    reranked_nodes.append(doc)

        return reranked_nodes
                    

class PairWiseReRanker(RerankerModel):
    
    def reranker(self, query_text: str, doc_a, doc_b):
        prompt = f"""Given the query: "{query_text}", which of the following documents is most relevant? Give one character answer: Either A or B.
        A: {doc_a.text}
        ////////////////////////////////////////////////////////////////////////////////
        B: {doc_b.text}
        """
        response = self.llm.complete(prompt)
        if 'B' in response:
            return 'B'
    def bubble_sort_docs(self, query_text: str, llama_index_docs):
        n = len(llama_index_docs)
        for i in range(n):
            for j in range(0, n-i-1):
                if self.reranker(query_text, llama_index_docs[j], llama_index_docs[j+1]) == 'B':
                    llama_index_docs[j], llama_index_docs[j+1] = llama_index_docs[j+1], llama_index_docs[j]
        return llama_index_docs

    def run(self, query_text: str, llama_index_docs):
        reranked_nodes = self.bubble_sort_docs(query_text, llama_index_docs)
        
        return reranked_nodes[:min(self.top_k, len(reranked_nodes))]
    
class ListWiseReranker(RerankerModel):
    def reranker(self, query_text: str, llama_index_docs):
        prompt = f"""Given the query: "{query_text}", Give the documents in decreasing order of relevance. You must give list of identifiers like A, B, C, D, E. 
        A: {llama_index_docs[0].text}
        ////////////////////////////////////////////////////////////////////////////////
        B: {llama_index_docs[1].text}
        ////////////////////////////////////////////////////////////////////////////////
        C: {llama_index_docs[2].text}
        ////////////////////////////////////////////////////////////////////////////////
        D: {llama_index_docs[3].text}
        ////////////////////////////////////////////////////////////////////////////////
        E: {llama_index_docs[4].text}
        """
        response = self.llm.complete(prompt)
        sorted_indices = []
        reranked_docs = []
        if 'A' in response:
            sorted_indices.append(response.index('A'))
        if 'B' in response:
            sorted_indices.append(response.index('B'))
        if 'C' in response:
            sorted_indices.append(response.index('C'))
        if 'D' in response:
            sorted_indices.append(response.index('D'))
        if 'E' in response:
            sorted_indices.append(response.index('E'))
        sorted_indices.sort()
        for index in sorted_indices:
            reranked_docs.append(llama_index_docs[index])
        return reranked_docs
            
    def run(self, query_text: str, llama_index_docs):
        reranked_nodes = self.reranker(query_text, llama_index_docs)
        return reranked_nodes[:min(self.top_k, len(reranked_nodes))]

