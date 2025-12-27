from pathlib import Path
from typing import Dict, Any
import json
import sys

"""
Extracción de texto desde PDF a ficheros intermedios:

  - data/interim/<nombre>.json  (páginas + texto)
  - data/interim/<nombre>.txt   (texto plano concatenado)

Usa PyMuPDF si está disponible; si no, cae a pdfminer.six.
"""

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def extract_with_pymupdf(pdf_path: Path) -> Dict[str, Any]:
    """Extrae páginas con PyMuPDF (rápido y robusto)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    return {
        "num_pages": len(doc),
        "pages": pages,
        "meta": dict(doc.metadata or {}),
    }


def extract_with_pdfminer(pdf_path: Path) -> Dict[str, Any]:
    """Alternativa basada en pdfminer.six si PyMuPDF no está disponible."""
    from pdfminer.high_level import extract_text

    text = extract_text(str(pdf_path))
    pages = [{"page": i + 1, "text": t} for i, t in enumerate(text.split("\f"))]
    return {
        "num_pages": len(pages),
        "pages": pages,
        "meta": {},
    }


def extract_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Elige el backend de extracción más adecuado disponible en runtime."""
    if fitz is not None:
        return extract_with_pymupdf(pdf_path)
    # Si PyMuPDF no está instalado, usamos pdfminer.six
    return extract_with_pdfminer(pdf_path)


def save_outputs(pdf_path: Path, result: Dict[str, Any]):
    """
    Guarda salidas en:
    - data/interim/<nombre>.json  (páginas + meta)
    - data/interim/<nombre>.txt   (todo el texto concatenado)
    """
    # Carpeta raíz del proyecto (…/asistente-turistico-cordoba)
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    out_base = pdf_path.stem
    out_json = data_dir / "interim" / f"{out_base}.json"
    out_txt = data_dir / "interim" / f"{out_base}.txt"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # JSON con páginas
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # TXT plano (todas las páginas concatenadas)
    joined = "\n\n".join([p["text"].strip() for p in result["pages"]])
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(joined)

    print(f"[OK] Wrote JSON: {out_json}")
    print(f"[OK] Wrote TXT : {out_txt}")


def main():
    """
    Uso:

        # Un solo PDF
        python -m cordoba_rag.ingest_pdf docs/cordoba/Guia_turistica_Cordoba.pdf

        # Una carpeta con muchos PDFs (caso típico del proyecto)
        python -m cordoba_rag.ingest_pdf docs/cordoba
    """
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m cordoba_rag.ingest_pdf <pdf_or_dir>\n"
            "Ejemplos:\n"
            "  python -m cordoba_rag.ingest_pdf docs/cordoba/Guia_turistica_Cordoba.pdf\n"
            "  python -m cordoba_rag.ingest_pdf docs/cordoba"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"[ERROR] No se han encontrado PDFs en {input_path}")
            sys.exit(1)
        print(f"[INFO] Procesando {len(pdf_files)} PDFs en {input_path}...")
    else:
        pdf_files = [input_path]

    for pdf_path in pdf_files:
        print(f"[INFO] Extrayendo texto de: {pdf_path}")
        result = extract_pdf(pdf_path)
        save_outputs(pdf_path, result)

    print("[INFO] Ingesta completada.")


if __name__ == "__main__":
    main()
