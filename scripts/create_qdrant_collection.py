from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, OptimizersConfigDiff, VectorParams
from cordoba_rag.embeddings import dim
import os

COLLECTION = os.getenv("QDRANT_COLLECTION", "cordoba_turismo")


def main():
    """
    Script de inicialización de colección en Qdrant.
    - Verifica si la colección existe.
    - Si no existe, la crea con el tamaño de vector adecuado y métrica coseno.
    - Crea también un índice de texto sobre el campo 'text'.
    """
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                          port=int(os.getenv("QDRANT_PORT", "6333")))

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"✅ La colección '{COLLECTION}' ya existe.")
    else:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim(), distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        )
        print(f"✅ Creada la colección '{COLLECTION}' con dim={dim()}")

    # Índice de texto sobre 'text'
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="text",
            field_schema="text",
        )
        print("✅ Índice de texto creado sobre el campo 'text'")
    except Exception as e:
        print(f"⚠️ No se pudo crear el índice (posiblemente ya exista): {e}")


if __name__ == "__main__":
    main()
