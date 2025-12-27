PY=python

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

extract:
	. .venv/bin/activate && PYTHONPATH=src $(PY) -m cordoba_rag.ingest_pdf docs/cordoba

chunk:
	. .venv/bin/activate && PYTHONPATH=src $(PY) -m cordoba_rag.chunking

create:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/create_qdrant_collection.py

reset:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/reset_collection.py

upsert:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/upsert_chunks.py

query:
	. .venv/bin/activate && PYTHONPATH=src $(PY) scripts/query_points.py

api:
	. .venv/bin/activate && PYTHONPATH=src uvicorn cordoba_rag.api_rag:app --reload --port 8000
