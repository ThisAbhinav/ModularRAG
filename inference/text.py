# from llama_index.readers.smart_pdf_loader import SmartPDFLoader
# import os
# import time
# from utils import loadllm, loadEmbeddingModel
# from llama_index.core import Settings
# from llama_index.core import SummaryIndex, Document
# from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
# import pandas as pd
# from utils import getllamaIndexDocument, addMetadata

# # os.environ["OLLAMA_BASE_URL"] = "http://10.195.100.5:11434"
# # llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# # try:
# #     langchain_docs = []
# #     for filename in os.listdir("data/themebooks/"):
# #         if filename.endswith(".pdf"):
# #             print(filename)
# #             file_path = os.path.join("data/themebooks/", filename)
# #             # loader = LLMSherpaFileLoader(
# #             #     file_path=file_path,
# #             #     new_indent_parser=True,
# #             #     apply_ocr=False,
# #             #     strategy="chunks",
# #             #     llmsherpa_api_url=llmsherpa_api_url,
# #             # )
# #             # documents = loader.load()
# #             pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)
# #             documents = pdf_loader.load_data(file_path)
# #             # documents = [doc.to_langchain_format() for doc in documents]
# #             langchain_docs.extend(documents)

# #             # texts = [doc.page_content for doc in langchain_docs]
# #             # metadata = [addMetadata(doc.metadata, "pdf") for doc in langchain_docs]
# #             # df = pd.DataFrame({"text": texts, "metadata": metadata})
# #             # llamaindex_docs = getllamaIndexDocument(df)
# #             break

# # except Exception as e:
# #     print(f"Failed to load PDF Data with error {e}")

# from llama_index.core import VectorStoreIndex


# Settings.embed_model = loadEmbeddingModel("huggingface", "all-mpnet-base-v2")
# index = VectorStoreIndex.from_documents(documents=langchain_docs, show_progress=True)
# query_engine = index.as_retriever(llm=loadllm("Ollama"))

# response = query_engine.retrieve("list all the deadlines")
# print(response)
# response = query_engine.retrieve("list all the deadlines in agribot theme")
# print(response)
# # response = query_engine.query("what is agribot?")
# # print(response)
