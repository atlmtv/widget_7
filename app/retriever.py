from app.embedder import Embedder

embedder = Embedder()
embedder.build_index('app/knowledge.txt')

def retrieve_relevant_chunks(query):
    return embedder.search(query)