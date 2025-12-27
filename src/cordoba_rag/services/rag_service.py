from __future__ import annotations

import os
import time
import unicodedata
from typing import Optional, List, Dict, Any
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText, PointStruct

from cordoba_rag.embeddings import embed_query, embed_passages

# Carga variables de entorno (.env)
load_dotenv()

# ─────── Configuración ────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en .env")
genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "cordoba_turismo")

# ──────── Helpers ────────
STOPWORDS_PLACEHOLDER = {"string", "none", "null", "undefined", "true", "false"}


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(
        (c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    )
    s = " ".join(s.split())
    return "" if s in STOPWORDS_PLACEHOLDER else s


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# ──────── Modelos ────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    filter_text: Optional[str] = None
    debug: bool = False


class UpsertRequest(BaseModel):
    texts: Optional[List[str]] = None
    text: Optional[str] = None
    source: str
    max_chars: int = 1200
    overlap: int = 120


class DeleteBySourceRequest(BaseModel):
    source: str


# ──────── Lógica principal RAG ────────
def ask(req: AskRequest) -> Dict[str, Any]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        qvec: np.ndarray = np.array(embed_query(req.question), dtype=float)

        qfilter = None
        if normalize_text(req.filter_text):
            qfilter = Filter(must=[FieldCondition(key="text", match=MatchText(text=req.filter_text))])

        initial_k = min(max(req.top_k * 3, req.top_k), 50)
        pts = client.query_points(
            collection_name=COLLECTION,
            query=qvec.tolist(),
            limit=initial_k,
            with_payload=True,
            with_vectors=True,
            query_filter=qfilter,
        ).points

        if not pts:
            return {
                "question": req.question,
                "answer": "No se encontró contexto relevante en la base vectorial.",
                "sources": [],
                "debug": {"filter_text_used": req.filter_text or None} if req.debug else None,
            }

        ranked: List[tuple] = []
        for p in pts:
            pvec = getattr(p, "vector", None)
            if pvec is None and hasattr(p, "vectors") and isinstance(p.vectors, dict):
                pvec = list(p.vectors.values())[0]
            sim = cos_sim(qvec, np.array(pvec, dtype=float)) if pvec is not None else 0.0
            ranked.append((p, sim))

        ranked.sort(key=lambda t: t[1], reverse=True)
        top = [p for (p, _) in ranked[: req.top_k]]

        snippets = [p.payload.get("text", "") for p in top if p.payload]
        context = "\n\n---\n\n".join(snippets)
        sources = [p.payload.get("source") for p in top if p.payload]

        prompt = f"""
Eres un guía turístico experto de la ciudad de Córdoba (España).
Responde en español de forma clara y EXCLUSIVAMENTE usando el CONTEXTO.
Si la información no está en el contexto, dilo explícitamente.

PREGUNTA:
{req.question}

CONTEXTO:
{context}
""".strip()

        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)

        out: Dict[str, Any] = {"question": req.question, "answer": resp.text, "sources": sources}
        if req.debug:
            out["debug"] = {
                "filter_text_used": req.filter_text or None,
                "hits": [
                    {"qdrant_score": getattr(p, "score", None), "chunk_id": (p.payload or {}).get("chunk_id")}
                    for p in top
                ],
            }
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en ask(): {e}")


def upsert(req: UpsertRequest) -> Dict[str, Any]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        if req.texts and len(req.texts) > 0:
            chunks = [t.strip() for t in req.texts if t and t.strip()]
        else:
            chunks = chunk_text(req.text or "", max_chars=req.max_chars, overlap=req.overlap)

        if not chunks:
            return {"status": "ok", "upserted": 0, "collection": COLLECTION, "source": req.source}

        vectors = embed_passages(chunks)
        now = int(time.time())

        points: List[PointStruct] = []
        for idx, (txt, vec) in enumerate(zip(chunks, vectors), start=1):
            payload = {
                "text": txt,
                "source": req.source,
                "page": None,
                "chunk_id": idx,
                "created_at": now,
            }
            points.append(PointStruct(id=uuid4().hex, vector=list(vec), payload=payload))

        client.upsert(collection_name=COLLECTION, points=points)
        return {"status": "ok", "upserted": len(points), "collection": COLLECTION, "source": req.source}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en upsert(): {e}")


def delete_by_source(req: DeleteBySourceRequest) -> Dict[str, Any]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qfilter = Filter(must=[FieldCondition(key="source", match=MatchText(text=req.source))])

        scroll_res = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qfilter,
            limit=1000,
        )
        deleted_estimate = len(scroll_res[0])

        client.delete(collection_name=COLLECTION, points_selector=qfilter)
        return {"status": "ok", "deleted_estimate": deleted_estimate, "source": req.source}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en delete_by_source(): {e}")


def stats() -> Dict[str, Any]:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        total = client.count(collection_name=COLLECTION, exact=True).count

        by_source: Dict[str, int] = {}
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=COLLECTION,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=next_page,
            )
            for p in points:
                src = (p.payload or {}).get("source", "unknown")
                by_source[src] = by_source.get(src, 0) + 1
            if not next_page:
                break

        return {"collection": COLLECTION, "total_points": total, "sources": by_source}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo stats(): {e}")


def health() -> Dict[str, Any]:
    ok_env = GEMINI_API_KEY is not None
    try:
        QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT).get_collection(COLLECTION)
        ok_qdrant = True
    except Exception:
        ok_qdrant = False

    return {
        "status": "ok" if (ok_env and ok_qdrant) else "degraded",
        "gemini_key": ok_env,
        "qdrant": ok_qdrant,
        "model": GEMINI_MODEL,
    }


def models() -> Dict[str, Any]:
    try:
        ms = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        return {"models": ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {e}")
