# 🏛️ Asistente Turístico de Córdoba (IA Conversacional + Voz + RAG)

Proyecto final de la asignatura **Herramientas de IA Clásica** del Máster en **IA, Cloud Computing y DevOps**.

Este repositorio contiene el desarrollo de un **asistente turístico inteligente sobre la ciudad de Córdoba**, capaz de interactuar mediante **texto y voz**, integrando técnicas de **RAG (Retrieval-Augmented Generation)**, una **base de datos vectorial**, y servicios de **IA generativa y voz**.

---

## 🎯 Objetivo del proyecto

El objetivo es construir un asistente que:

- Responda preguntas turísticas en **lenguaje natural**.
- Utilice **únicamente información previamente indexada** (evitando alucinaciones).
- Permita interacción por **texto y voz**.
- Esté disponible vía **API REST**, **Telegram** y **Dialogflow**.
- Integre servicios reales de IA clásica y moderna.

---

## 🚀 Funcionalidades principales

- 🔎 **RAG (Retrieval-Augmented Generation)** sobre documentación local.
- 💬 Consultas por **texto**.
- 🎙️ Consultas por **voz** (STT + TTS).
- 🤖 Integración con **Telegram** (texto y notas de voz).
- 🧩 Integración con **Dialogflow ES** mediante webhook.
- 🗂️ Persistencia vectorial local con **Qdrant**.
- 🐳 Ejecución en entorno local con **Docker y Python**.

---

## 🧠 Arquitectura general

**Flujo de texto**
1. Entrada del usuario.
2. Generación de embedding.
3. Recuperación semántica en Qdrant.
4. Construcción de contexto.
5. Generación de respuesta con Gemini.

**Flujo de voz**
1. Audio → Speech-to-Text (Azure).
2. Texto → pipeline RAG.
3. Respuesta → Text-to-Speech (Azure).
4. Devolución de audio WAV.

---

## 🧰 Tecnologías utilizadas

- **Python 3.10+**
- **FastAPI**
- **Google Gemini** (`gemini-2.5-flash`)
- **Qdrant** (Vector DB)
- **Sentence Transformers**
  - `intfloat/multilingual-e5-small` (optimizado para español)
- **Azure Cognitive Services – Speech**
- **Telegram Bot API**
- **Dialogflow ES**
- **Docker & Docker Compose**
- **ngrok**

---

## 📁 Estructura del proyecto

```text
ASISTENTE-TURISTICO-CORDOBA
├─ data/
│  ├─ interim/               # Texto y JSON intermedio
│  ├─ processed/             # Chunks finales (JSONL)
│  └─ audio/                 # Audios de prueba
├── docs/
│   ├── cordoba/             # PDFs originales
│   └── memoria/             # Memoria del proyecto
├─ qdrant_config/            # Configuración de Qdrant
├─ qdrant_data/              # Persistencia local
├── scripts/                 # Scripts de ingesta y pruebas
│   ├── stt_file_test.py
│   ├── tts_test.py
│   └── ingest_chunks.py
├─ src/cordoba_rag/
│  ├─ api_rag.py             # API principal FastAPI
│  ├─ telegram_webhook.py    # Webhook de Telegram
│  ├─ api.py                 # Punto de entrada alternativo
│  ├─ chunking.py            # Lógica de troceado
│  ├─ embeddings.py          # Cálculo de embeddings
│  ├─ ingest_pdf.py          # Extracción de texto
│  └─ services/
│     ├─ rag_service.py      # Lógica RAG (ask, stats, upsert…)
│     ├─ voice_service.py    # STT y TTS con Azure
│     └─ __init__.py
├─ docker-compose.yaml
├─ Makefile
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## ⚙️ Configuración

1. Crear el archivo de entorno:

```bash
cp .env.example .env
```

2. Completar las variables:

```env
# Gemini
GEMINI_API_KEY=TU_API_KEY
GEMINI_MODEL=gemini-2.5-flash

# Embeddings
EMBEDDINGS_MODEL=intfloat/multilingual-e5-small

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=cordoba_turismo

# Telegram
TELEGRAM_BOT_TOKEN=TU_TOKEN

# Azure Speech
SPEECH_KEY=TU_SPEECH_KEY
SPEECH_REGION=swedencentral
```

---

## 🐳 Qdrant (Vector Database)

```bash
docker compose up -d
```

Dashboard:
```
http://localhost:6333/dashboard
```

---

## 📥 Ingesta de documentos

```bash
make extract
make chunk
make upsert
```

---

## ▶️ Arranque de la API

```bash
make api
```

- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## 🔌 Endpoints principales

### Texto
```http
POST /ask
```

```json
{ "question": "¿Qué puedo visitar en Córdoba en 3 días?" }
```

### Voz
```http
POST /voice
```

Audio recomendado:
- WAV
- 16 kHz
- Mono

Ejemplo:
```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

---

## 🤖 Integración con Telegram

- Mensajes de texto → respuesta en texto.
- Notas de voz → respuesta en audio.

Configuración del webhook:

```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook"   -d "url=https://TU_SUBDOMINIO.ngrok-free.dev/telegram/webhook"
```

---

## 🧠 Integración con Dialogflow

- Intents con **Enable webhook call**.
- Webhook configurado hacia:
```
/fulfillment
```

---

## 📄 Memoria del proyecto

La memoria completa se encuentra en:

```
docs/memoria/Memoria_Asistente_Turistico_Cordoba.pdf
```

---

## ✍️ Autora

**Ana Belén Ballesteros Redondo**  
Máster en IA, Cloud Computing y DevOps

