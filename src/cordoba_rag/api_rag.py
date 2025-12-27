from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

from cordoba_rag.telegram_webhook import router as telegram_router
from cordoba_rag.services import (
    AskRequest, UpsertRequest, DeleteBySourceRequest,
    ask, upsert, delete_by_source, stats, health, models,
    stt_from_wav, tts_to_wav,
)

load_dotenv()

app = FastAPI(
    title="Asistente turístico de Córdoba (RAG + Gemini)",
    version="1.0",
    description=(
        "API RAG que actúa como asistente turístico de la ciudad de Córdoba.\n\n"
        "- Usa una base vectorial en Qdrant con información de folletos y guías turísticas.\n"
        "- Utiliza Gemini para generar respuestas basadas solo en el contexto recuperado."
    ),
)

app.include_router(telegram_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Asistente turístico de Córdoba (RAG + Gemini) up", "docs": "/docs"}


@app.get("/health")
def api_health():
    return health()


@app.get("/models")
def api_models():
    return models()


@app.get("/stats")
def api_stats():
    return stats()


@app.post("/ask")
def api_ask(req: AskRequest):
    return ask(req)


@app.post("/upsert")
def api_upsert(req: UpsertRequest):
    return upsert(req)


@app.post("/delete_by_source")
def api_delete_by_source(req: DeleteBySourceRequest):
    return delete_by_source(req)


@app.post("/fulfillment")
async def fulfillment(req: Request):
    body = await req.json()
    user_text = ((body.get("queryResult") or {}).get("queryText") or (body.get("text") or "")).strip()

    if not user_text:
        return {"fulfillmentText": "No he recibido texto. ¿Puedes repetir la pregunta?"}

    rag_out = await run_in_threadpool(ask, AskRequest(question=user_text, top_k=5, debug=False))
    answer = rag_out.get("answer", "No he encontrado información relevante en mi base de conocimiento.")
    return {"fulfillmentText": answer}


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .wav (recomendado 16kHz mono PCM).")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.wav")
        out_path = os.path.join(tmpdir, "output.wav")

        content = await audio.read()
        with open(in_path, "wb") as f:
            f.write(content)

        user_text = stt_from_wav(in_path).strip()

        try:
            rag_out = await run_in_threadpool(ask, AskRequest(question=user_text, top_k=5, debug=False))
            reply_text = rag_out.get("answer", "No he podido generar una respuesta con la información disponible.")
        except Exception as e:
            print("ERROR en /voice llamando a ask():", repr(e))
            reply_text = "Lo siento, ha ocurrido un error al consultar la información turística. ¿Puedes repetir la pregunta?"

        reply_text = reply_text.strip()
        if len(reply_text) > 600:
            reply_text = reply_text[:600].rsplit(" ", 1)[0] + "..."

        tts_to_wav(reply_text, out_path)

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

    return Response(content=audio_bytes, media_type="audio/wav")
