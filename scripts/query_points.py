from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from cordoba_rag.embeddings import embed
import os

COLLECTION = os.getenv("QDRANT_COLLECTION", "cordoba_turismo")


def main():
    """
    Script de prueba rápida para comprobar que la búsqueda semántica funciona.
    """
    q = "¿Qué es la ruta de Manolete en Córdoba?"
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                          port=int(os.getenv("QDRANT_PORT", "6333")))
    vec = embed([q])[0]

    res = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=3,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(should=[]),
    ).points

    for i, r in enumerate(res, 1):
        print(f"[{i}] score={r.score:.4f}")
        print(r.payload.get("text", "")[:400], "\n---\n")


if __name__ == "__main__":
    main()
