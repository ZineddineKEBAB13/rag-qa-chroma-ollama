from create_chunks import create_sentences , Chunks_creator
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

doc1 = create_sentences(r"data\doc1.txt")
doc2 = create_sentences(r"data\doc2.txt")


chunks1 = Chunks_creator(doc1).create_chunks()
chunks2 = Chunks_creator(doc2).create_chunks()


model = SentenceTransformer("sentence-transformers/msmarco-bert-base-dot-v5")


embeddings1 = model.encode(chunks1,show_progress_bar = False)
embeddings2 = model.encode(chunks2,show_progress_bar = False)

data = np.concatenate((embeddings1,embeddings2))

chroma_client = chromadb.PersistentClient(path ="./db_ai_search")
collection = chroma_client.get_or_create_collection(name = "AI_and_Climate_docs")

ids = [f"doc1_{i}" for i in range(len(chunks1))] + [f"doc2_{i}" for i in range(len(chunks2))]
metadatas = [{"source":"doc1.txt", "chunk":i} for i in range(len(chunks1))] + [{"source":"doc2.txt", "chunk":i} for i in range(len(chunks2))]

collection.add(ids = ids , documents = chunks1+chunks2 , embeddings = data.tolist(), metadatas = metadatas)




































