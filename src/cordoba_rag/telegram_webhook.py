from __future__ import annotations

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from cordoba_rag.services import AskRequest, ask, stt_from_wav, tts_to_wav

router = APIRouter(prefix="/telegram", tags=["telegram"])

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en .env")

TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


def tg_post(method: str, data: Dict[str, Any] | None = None, files: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{TG_API}/{method}"
    r = requests.post(url, data=data, files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def tg_get_file_url(file_id: str) -> str:
    info = tg_post("getFile", data={"file_id": file_id})
    file_path = info.get("result", {}).get("file_path")
    if not file_path:
        raise HTTPException(status_code=500, detail="Telegram getFile no devolvió file_path")
    return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"


def download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ogg_to_wav_16k_mono(in_ogg: Path, out_wav: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_ogg),
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg error: {p.stderr[-400:]}")


@router.post("/webhook")
async def telegram_webhook(req: Request):
    update = await req.json()

    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")

    if not chat_id:
        return {"ok": True}

    # 1) Texto normal
    text: Optional[str] = msg.get("text")
    if text:
        rag_out = await run_in_threadpool(ask, AskRequest(question=text, top_k=5, debug=False))
        answer = rag_out.get("answer", "No he encontrado información relevante.")
        tg_post("sendMessage", data={"chat_id": chat_id, "text": answer})
        return {"ok": True}

    # 2) Nota de voz / audio
    voice = msg.get("voice")
    audio = msg.get("audio")

    file_id = None
    if voice and isinstance(voice, dict):
        file_id = voice.get("file_id")
    elif audio and isinstance(audio, dict):
        file_id = audio.get("file_id")

    if not file_id:
        tg_post("sendMessage", data={"chat_id": chat_id, "text": "Envíame texto o una nota de voz 🙂"})
        return {"ok": True}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_ogg = tmp / "input.ogg"
        in_wav = tmp / "input.wav"
        out_wav = tmp / "output.wav"

        file_url = tg_get_file_url(file_id)
        download_file(file_url, in_ogg)

        ogg_to_wav_16k_mono(in_ogg, in_wav)

        user_text = stt_from_wav(str(in_wav)).strip()

        rag_out = await run_in_threadpool(ask, AskRequest(question=user_text, top_k=5, debug=False))
        reply_text = rag_out.get("answer", "No he encontrado información relevante.").strip()

        if len(reply_text) > 900:
            reply_text = reply_text[:900].rsplit(" ", 1)[0] + "..."

        tts_to_wav(reply_text, str(out_wav))

        with open(out_wav, "rb") as f:
            tg_post(
                "sendAudio",
                data={"chat_id": chat_id},
                files={"audio": ("respuesta.wav", f, "audio/wav")},
            )

    return {"ok": True}
