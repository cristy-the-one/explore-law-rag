import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser

from src.config import CHUNK_SIZES

logger = logging.getLogger(__name__)

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)


def _is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Heuristic: if average extracted text per page < 200 chars, assume scanned.
    Falls back to True (trigger Docling) on any fitz error.
    """
    try:
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        if num_pages == 0:
            doc.close()
            return True

        total_text = sum(len(page.get_text("text").strip()) for page in doc)
        doc.close()
        return (total_text / num_pages) < 200
    except Exception:
        return True


class LegalTextSplitter:
    """
    Splits Romanian legal text on strong structural markers before any
    hierarchical chunking. Respects article / alineat boundaries.
    """

    _PATTERN = re.compile(
        r"((?:ARTICOLUL|Art\.|ALIN\.\s*\([0-9a-z]+\)|Capitolul|Secțiunea|Titlul|Anexa)\s)",
        re.IGNORECASE,
    )
    _MIN_CHUNK_LEN = 50

    def split_text(self, text: str) -> list[str]:
        parts = self._PATTERN.split(text)
        result: list[str] = []

        preamble = parts[0].strip() if parts else ""
        if len(preamble) >= self._MIN_CHUNK_LEN:
            result.append(preamble)

        for i in range(1, len(parts), 2):
            marker = parts[i]
            content = parts[i + 1] if (i + 1) < len(parts) else ""
            chunk = (marker + content).strip()
            if len(chunk) >= self._MIN_CHUNK_LEN:
                result.append(chunk)

        return result if result else [text]


def _extract_metadata_regex(text: str) -> dict:
    """Fast path: regex for well-formed Monitorul Oficial text."""
    law_match = re.search(r"(?:LEGE|OUG|HG|ORDIN|COD)\s*nr\.?\s*(\d+)[/\s-](\d{4})", text, re.I)
    law_nr = f"{law_match.group(1)}/{law_match.group(2)}" if law_match else None

    date_match = re.search(r"(?:intrat în vigoare|publicat[ă]?).*?(\d{1,2}\.\d{1,2}\.\d{4})", text, re.I)
    effective_date = date_match.group(1) if date_match else None

    return {"law_nr": law_nr, "effective_date": effective_date}


def _extract_metadata_llm(text: str, llm) -> dict:
    """
    Fallback: one LLM call on first 1000 chars.
    Receives llm_ro — passed from pipeline.py to avoid circular import.
    """
    prompt = f"""Din textul următor extrage:
1. Numărul și anul actului normativ (ex: \"123/2024\") sau \"unknown\"
2. Data intrării în vigoare (format zz.ll.aaaa) sau \"unknown\"
Răspunde DOAR cu JSON: {{\"law_nr\": \"...\", \"effective_date\": \"...\"}}
Fără text suplimentar, fără markdown.

Text: {text[:1000]}"""
    try:
        response = llm.complete(prompt).text.strip()
        response = re.sub(r"^```[a-z]*\n?", "", response)
        response = re.sub(r"\n?```$", "", response)
        data = json.loads(response)
        return {
            "law_nr": str(data.get("law_nr", "unknown")),
            "effective_date": str(data.get("effective_date", "unknown")),
        }
    except Exception as exc:
        logger.warning("LLM metadata fallback failed: %s", exc)
        return {"law_nr": "unknown", "effective_date": "unknown"}


def parse_legal_document(file_path: Path, llm=None) -> tuple[Document, dict]:
    """
    Parse a PDF into a LlamaIndex Document with clean string metadata.
    """
    md_text = None

    if not _is_scanned_pdf(file_path):
        try:
            doc = fitz.open(str(file_path))
            md_text = "\n\n".join(page.get_text("text") for page in doc)
            doc.close()
            if len(md_text.strip()) < 500:
                md_text = None
            else:
                logger.info("PyMuPDF fast path: %s", file_path.name)
        except Exception as exc:
            logger.warning("PyMuPDF failed for %s: %s", file_path.name, exc)
            md_text = None

    if md_text is None:
        logger.info("Docling fallback: %s", file_path.name)
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        md_text = result.document.export_to_markdown()

    meta = _extract_metadata_regex(md_text)

    if (meta["law_nr"] is None or meta["effective_date"] is None) and llm is not None:
        logger.info("Regex metadata failed for %s — using LLM fallback", file_path.name)
        fallback = _extract_metadata_llm(md_text, llm)
        if meta["law_nr"] is None:
            meta["law_nr"] = fallback["law_nr"]
        if meta["effective_date"] is None:
            meta["effective_date"] = fallback["effective_date"]

    metadata = {
        "filename": file_path.stem,
        "law_nr": meta["law_nr"] or "unknown",
        "effective_date": meta["effective_date"] or "unknown",
        "source_type": "romanian_legal",
    }

    return Document(text=md_text, metadata=metadata), metadata


_JSON_NOISE_PATTERNS = [
    r"×\s*Inchide\s+Versiune Beta.*$",
    r"×\s*Inchide\s+Datorita faptului.*$",
    r"Reveniti in topul paginii.*$",
    r"Forma printabilă\s*$",
    r"Ce browser sa aleg\?.*$",
]
_JSON_NOISE_RE = re.compile("|".join(_JSON_NOISE_PATTERNS), re.IGNORECASE | re.DOTALL)


def _clean_json_text(raw_text: str) -> str:
    """
    Remove browser UI noise from legislatie.just.ro text_complet field.
    Noise always appears in the tail — we strip from the first match onward.
    """
    cleaned = _JSON_NOISE_RE.sub("", raw_text)
    return cleaned.strip()


def parse_json_document(file_path: Path) -> tuple[Document, dict]:
    """
    Parse a legislatie.just.ro JSON file into a LlamaIndex Document.
    """
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse JSON %s: %s", file_path.name, exc)
        raise

    tip_act = str(raw.get("tip_act", "")).strip()
    numar = str(raw.get("numar", "0")).strip()
    doc_id = str(raw.get("id", file_path.stem)).strip()

    if numar and numar != "0":
        law_nr = f"{tip_act} {numar}".strip()
    else:
        law_nr = f"{tip_act} [{doc_id}]".strip() if tip_act else doc_id

    data_vigoare_raw = str(raw.get("data_vigoare", "")).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", data_vigoare_raw):
        parts = data_vigoare_raw.split("-")
        effective_date = f"{parts[2]}.{parts[1]}.{parts[0]}"
    else:
        effective_date = data_vigoare_raw or "unknown"

    metadata = {
        "filename": file_path.stem,
        "law_nr": law_nr or "unknown",
        "effective_date": effective_date,
        "source_type": "romanian_legal",
        "tip_act": tip_act,
        "emitent": str(raw.get("emitent", "")).strip(),
        "in_vigoare": str(raw.get("in_vigoare", "true")),
        "source_url": str(raw.get("link", "")).strip(),
        "doc_id": doc_id,
    }

    raw_text = raw.get("text_complet", "") or raw.get("text_snippet", "")
    if not raw_text.strip():
        logger.warning("Empty text_complet in %s — document will be low-quality", file_path.name)
        raw_text = raw.get("titlu", "")

    clean_text = _clean_json_text(raw_text)

    titlu = str(raw.get("titlu", "")).strip()
    if titlu and not clean_text.startswith(titlu[:50]):
        clean_text = f"{titlu}\n\n{clean_text}"

    return Document(text=clean_text, metadata=metadata), metadata
