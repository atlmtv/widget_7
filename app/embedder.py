import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []

    def split_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_index(self, text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.text_chunks = self.split_text(text)
        embeddings = self.model.encode(self.text_chunks)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        return [self.text_chunks[i] for i in I[0]]