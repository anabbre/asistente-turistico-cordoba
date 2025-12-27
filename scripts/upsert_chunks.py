from pathlib import Path
import json
import uuid
from typing import Dict, List
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from cordoba_rag.embeddings import embed

COLLECTION = os.getenv("QDRANT_COLLECTION", "cordoba_turismo")


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def det_uuid(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text.strip()))


def main() -> None:
    src = Path("data/processed/chunks.jsonl")
    if not src.exists():
        raise SystemExit("Falta data/processed/chunks.jsonl (ejecuta chunking primero).")

    docs = load_jsonl(src)
    texts = [d["text"] for d in docs]
    vectors = embed(texts)

    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                          port=int(os.getenv("QDRANT_PORT", "6333")))
    points: List[PointStruct] = []

    for d, vec in zip(docs, vectors, strict=True):
        hid = det_uuid(d["text"])
        payload = {k: v for k, v in d.items() if k != "id"}
        payload["hash"] = hid
        points.append(
            PointStruct(
                id=hid,
                vector=vec,
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upsert OK: {len(points)}")


if __name__ == "__main__":
    main()
