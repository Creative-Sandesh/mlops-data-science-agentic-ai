from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings


class RAGVectorStore:
    """In-memory Qdrant collection + sentence-transformers embeddings (W15 design).

    W17 fixes: `client.search` was removed from qdrant-client (-> `query_points`), and the
    collection is (re)built without the deprecated `recreate_collection`.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        # Using in-memory Qdrant for easy setup, or connect via HTTP
        self.client = QdrantClient(location=":memory:")
        self.collection_name = "knowledge_base"
        self._next_id = 0
        self._init_collection()

    def _init_collection(self):
        vector_size = self.embedder.get_sentence_embedding_dimension()
        if vector_size is None:
            raise ValueError("Embedding model returned None for vector size.")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self._next_id = 0

    def add_documents(self, chunks: list[str]):
        embeddings = self.embedder.encode(chunks)
        points = [
            PointStruct(id=self._next_id + i, vector=emb.tolist(), payload={"text": chunk})
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        self._next_id += len(points)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        top_k = top_k or settings.TOP_K
        query_vector = self.embedder.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k
        ).points
        return [hit.payload["text"] for hit in results if hit.payload and "text" in hit.payload]
