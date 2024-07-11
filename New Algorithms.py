from dataingestion import MarkDownDataIngestion
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
data_loader = MarkDownDataIngestion()
docs = data_loader.load_data('outputs_md')[2]
doc_string = ""
for doc in docs:
    doc_string += doc.text + " "

def create_chunks(doc_string, chunk_size):
    words = doc_string.split()
    chunks = []
    current_chunk = ""
    current_start = 0

    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:  # +1 for the space
            current_chunk += word + " "
        else:
            current_end = current_start + len(current_chunk)
            chunks.append(Document(text=current_chunk.strip(), metadata={'start': current_start, 'end': current_end, 'chunk_size': chunk_size}))
            current_start = current_end + 1
            current_chunk = word + " "

    if current_chunk:  # Add the last chunk if it exists
        current_end = current_start + len(current_chunk)
        chunks.append(Document(text=current_chunk.strip(), metadata={'start': current_start, 'end': current_end, 'chunk_size': chunk_size}))
        
    return chunks

chunks_1000 = create_chunks(doc_string, 1000)
chunks_500 = create_chunks(doc_string, 500)
chunks_250 = create_chunks(doc_string, 250)
chunks_100 = create_chunks(doc_string, 100)
all_chunks = chunks_1000 + chunks_500 + chunks_250 + chunks_100
index = VectorStoreIndex.from_documents(all_chunks)






print("Index created with", len(index), "documents")