from __future__ import annotations

import os
import time
import unicodedata
from typing import Optional, List, Dict, Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchText, PointStruct
)

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
COLLECTION   = os.getenv("QDRANT_COLLECTION", "audiovisual_2025")

# Import diferido (solo se carga SentenceTransformer si se necesita)
from informe_sector_audiovisual_2025.embeddings import embed_query, embed_passages  # noqa: E402

# Inicializa la API
app = FastAPI(
    title="RAG Audiovisual 2025 con Gemini",
    version="1.0",
    description=(
        "API para consultas y actualización del **Informe del Sector Audiovisual 2025**.\n\n"
        "Usa Qdrant como base vectorial + re-ranking coseno y Gemini para responder solo con el contexto recuperado."
    ),
)

# CORS abierto 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ──────── Helpers ────────
STOPWORDS_PLACEHOLDER = {"string", "none", "null", "undefined", "true", "false"}

def normalize_text(s: Optional[str]) -> str:
    """Minúsculas, sin acentos, sin símbolos raros y espacios normalizados."""
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(
        (c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    )
    s = " ".join(s.split())
    return "" if s in STOPWORDS_PLACEHOLDER else s

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similitud coseno segura entre dos vectores."""
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    """Divide texto en trozos con solape (caracteres)."""
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


# ──────── Modelos de entrada/salida ────────
class AskRequest(BaseModel):
    """Petición para /ask."""
    question: str
    top_k: int = 5
    filter_text: Optional[str] = None
    debug: bool = False

class UpsertRequest(BaseModel):
    # O bien pasas 'texts' (lista de trozos ya listos)...
    texts: Optional[List[str]] = None
    # ...o pasas 'text' (un texto largo para trocear aquí).
    text: Optional[str] = None
    source: str
    max_chars: int = 1200
    overlap: int = 120

class DeleteBySourceRequest(BaseModel):
    """Petición para /delete_by_source."""
    source: str


# ──────── Endpoints ────────
@app.get("/")
def root():
    """Ping simple de la API (muestra enlace a documentación)."""
    return {"message": "RAG Audiovisual 2025 up", "docs": "/docs"}

@app.get("/health")
def health():
    """Comprueba conexión con Gemini y Qdrant."""
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

@app.get("/models")
def models():
    """Lista los modelos de Gemini disponibles para generateContent."""
    try:
        ms = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        return {"models": ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {e}")


# RAG principal (/ask)
@app.post("/ask")
def ask(req: AskRequest):
    """
    Endpoint principal de QA (RAG).

    Flujo:
    1. Calcula embedding de la pregunta.
    2) Recuperación en Qdrant (query_points) con sobre-muestreo.
    3. Reordena localmente por similitud coseno.
    4. Construye contexto textual.
    5. Envía prompt a Gemini y devuelve respuesta + fuentes.
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # 1) Embedding de la pregunta
        qvec: np.ndarray = np.array(embed_query(req.question), dtype=float)

        # 2) Filtro de texto robusto (opcional)
        qfilter = None
        if normalize_text(req.filter_text):
            qfilter = Filter(must=[FieldCondition(key="text", match=MatchText(text=req.filter_text))])

        # 3) Recuperación (sobre-muestreo + con vector para re-ranking)
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

        # 4) Re-ranking local por coseno
        ranked: List[tuple] = []
        for p in pts:
            pvec = getattr(p, "vector", None)
            if pvec is None and hasattr(p, "vectors") and isinstance(p.vectors, dict):
                pvec = list(p.vectors.values())[0]
            sim = cos_sim(qvec, np.array(pvec, dtype=float)) if pvec is not None else 0.0
            ranked.append((p, sim))
        ranked.sort(key=lambda t: t[1], reverse=True)
        top = [p for (p, _) in ranked[:req.top_k]]

        # 5) Contexto + fuentes
        snippets = [p.payload.get("text", "") for p in top if p.payload]
        context = "\n\n---\n\n".join(snippets)
        sources = [p.payload.get("source") for p in top if p.payload]

        # 6) Prompt a Gemini
        prompt = f"""
Eres un analista del sector audiovisual.
Responde en español de forma breve y **EXCLUSIVAMENTE** usando el CONTEXTO.
Si la información no está en el contexto, indícalo claramente.

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
                    {
                        "qdrant_score": getattr(p, "score", None),
                        "chunk_id": (p.payload or {}).get("chunk_id"),
                    }
                    for p in top
                ],
            }
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en /ask: {e}")


# Upsert (inserta/actualiza) datos en Qdrant
@app.post("/upsert")
def upsert(req: UpsertRequest):
    """Inserta o actualiza embeddings en Qdrant a partir de texto o fragmentos."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Construir lista de trozos
        if req.texts and len(req.texts) > 0:
            chunks = [t.strip() for t in req.texts if t and t.strip()]
        else:
            chunks = chunk_text(req.text or "", max_chars=req.max_chars, overlap=req.overlap)

        if not chunks:
            return {"status": "ok", "upserted": 0, "collection": COLLECTION, "source": req.source}

        # Embeddings por lotes
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
        raise HTTPException(status_code=500, detail=f"Error en /upsert: {e}")


@app.post("/delete_by_source")
def delete_by_source(req: DeleteBySourceRequest):
    """Elimina todos los puntos cuyo payload.source coincide con la fuente dada."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qfilter = Filter(must=[FieldCondition(key="source", match=MatchText(text=req.source))])

        # Conteo previo opcional (para informar)
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
        raise HTTPException(status_code=500, detail=f"Error en /delete_by_source: {e}")



@app.get("/stats")
def stats():
    """Muestra métricas básicas: total de puntos y distribución por 'source'."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # total puntos (count de Qdrant)
        total = client.count(collection_name=COLLECTION, exact=True).count

        # desglose por source (scroll por lotes)
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
        raise HTTPException(status_code=500, detail=f"Error obteniendo stats: {e}")

