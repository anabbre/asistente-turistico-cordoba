from __future__ import annotations  # ← debe ser la primera línea

import os

# Fuerza desactivar XET y hf_transfer (anula lo del shell)
os.environ["HUGGINGFACE_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from typing import List
from sentence_transformers import SentenceTransformer

# Modelo por defecto – multilingüe optimizado para ES (dim=384)
DEFAULT_MODEL = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-small")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Carga perezosa (singleton) del modelo de embeddings."""
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_MODEL)
    return _model


def _needs_prefix(model_name: str) -> bool:
    """
    Modelos tipo e5 / GTE usan prefijos 'query:' / 'passage:' para mejorar rendimiento.
    """
    name = model_name.lower()
    return ("e5" in name) or ("gte" in name)


def embed_texts(texts: List[str], kind: str = "passage") -> List[List[float]]:
    """
    Embeddings normalizados para una lista de textos.
    kind: 'query' para consultas, 'passage' para documentos.
    """
    model = get_model()
    if _needs_prefix(DEFAULT_MODEL):
        prefix = "query: " if kind == "query" else "passage: "
        texts = [prefix + (t or "") for t in texts]
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_passages(texts: List[str]) -> List[List[float]]:
    """Embeddings para pasajes/documentos."""
    return embed_texts(texts, kind="passage")


def embed_query(text: str) -> List[float]:
    """Embedding para una sola consulta."""
    return embed_texts([text], kind="query")[0]


def dim() -> int:
    """Dimensionalidad del espacio de embeddings."""
    return int(get_model().get_sentence_embedding_dimension())


# Compatibilidad hacia atrás (por si algún script usa 'embed') 
def embed(texts: List[str]) -> List[List[float]]:
    """Alias legacy: trata todos los textos como 'passage'."""
    return embed_passages(texts)
