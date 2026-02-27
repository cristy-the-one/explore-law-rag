## Plan Update: v1.4 → v1.5 — JSON Document Support

### What's in the JSON and what it means for the plan

The JSON format (`legislatie.just.ro`) provides structured metadata that PDFs don't — `tip_act`, `numar`, `data_vigoare`, `emitent`, `in_vigoare` are already clean fields. This means the entire regex + LLM metadata fallback chain is bypassed for JSON files. The full legal text lives in `text_complet`, which needs one cleaning pass to strip browser UI noise appended at the end (`× Inchide Versiune Beta...`, browser warning blocks). The `+` separators between chapters/articles are compatible with `LegalTextSplitter` since it already splits on `Capitolul`, `Articolul`, etc.

---

## Changelog v1.4 → v1.5

| #   | File          | Change                                    | Why                                                                            |
| --- | ------------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| 1   | `parsers.py`  | Add `_clean_json_text()`                  | Strips browser UI noise from `text_complet` tail                               |
| 2   | `parsers.py`  | Add `parse_json_document()`               | Extracts metadata directly from JSON fields — no regex, no LLM fallback needed |
| 3   | `pipeline.py` | Collect `*.pdf` + `*.json` from `RAW_DIR` | Unified file loop with per-extension routing                                   |
| 4   | `pipeline.py` | Route to correct parser by file suffix    | `parse_legal_document()` for PDF, `parse_json_document()` for JSON             |

**Zero new dependencies.** `json` is stdlib. Everything else is existing imports.

All v1.3 patches and v1.4 enhancements are preserved exactly.

---

## `src/parsers.py` ← CHANGED in v1.5

Full file — all previous content preserved, two additions at the bottom:

````python
import re
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser

from src.config import CHUNK_SIZES

logger = logging.getLogger(__name__)

# Instantiated once at module level
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)


# ── Stage 1: Hybrid PDF extraction ───────────────────────────────────────────

def _is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Heuristic: if average extracted text per page < 200 chars, assume scanned.
    Falls back to True (trigger Docling) on any fitz error.
    """
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return True
        total_text = sum(len(page.get_text("text").strip()) for page in doc)
        doc.close()
        return (total_text / len(doc)) < 200
    except Exception:
        return True


# ── Stage 2: Legal-aware structural pre-chunking ─────────────────────────────

class LegalTextSplitter:
    """
    Splits Romanian legal text on strong structural markers before any
    hierarchical chunking. Respects article / alineat boundaries so
    LawMate receives clean, complete legal units.

    Markers: ARTICOLUL, Art., ALIN.(n), Capitolul, Secțiunea, Titlul, Anexa

    Usage (in pipeline.py):
        splitter = LegalTextSplitter()
        pre_chunks = splitter.split_text(doc.text)
        sub_docs = [Document(text=c, metadata=doc.metadata) for c in pre_chunks]
        nodes = node_parser.get_nodes_from_documents(sub_docs)
    """

    _PATTERN = re.compile(
        r'((?:ARTICOLUL|Art\.|ALIN\.\s*\([0-9a-z]+\)'
        r'|Capitolul|Secțiunea|Titlul|Anexa)\s)',
        re.IGNORECASE
    )
    _MIN_CHUNK_LEN = 50

    def split_text(self, text: str) -> list[str]:
        parts = self._PATTERN.split(text)
        result = []

        preamble = parts[0].strip()
        if len(preamble) >= self._MIN_CHUNK_LEN:
            result.append(preamble)

        for i in range(1, len(parts), 2):
            marker  = parts[i]
            content = parts[i + 1] if (i + 1) < len(parts) else ""
            chunk   = (marker + content).strip()
            if len(chunk) >= self._MIN_CHUNK_LEN:
                result.append(chunk)

        return result if result else [text]


# ── PDF metadata extraction ───────────────────────────────────────────────────

def _extract_metadata_regex(text: str) -> dict:
    """Fast path: regex for well-formed Monitorul Oficial text."""
    law_match = re.search(
        r"(?:LEGE|OUG|HG|ORDIN|COD)\s*nr\.?\s*(\d+)[/\s-](\d{4})",
        text, re.I
    )
    law_nr = f"{law_match.group(1)}/{law_match.group(2)}" if law_match else None

    date_match = re.search(
        r"(?:intrat în vigoare|publicat[ă]?).*?(\d{1,2}\.\d{1,2}\.\d{4})",
        text, re.I
    )
    effective_date = date_match.group(1) if date_match else None

    return {"law_nr": law_nr, "effective_date": effective_date}


def _extract_metadata_llm(text: str, llm) -> dict:
    """
    Fallback: one LLM call on first 1000 chars.
    Used when regex fails (scanned PDFs, amendments, atypical formatting).
    Receives llm_ro — passed from pipeline.py to avoid circular import.
    """
    prompt = f"""Din textul următor extrage:
1. Numărul și anul actului normativ (ex: "123/2024") sau "unknown"
2. Data intrării în vigoare (format zz.ll.aaaa) sau "unknown"
Răspunde DOAR cu JSON: {{"law_nr": "...", "effective_date": "..."}}
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
    except Exception as e:
        logger.warning(f"LLM metadata fallback failed: {e}")
        return {"law_nr": "unknown", "effective_date": "unknown"}


# ── PDF parse entry point ─────────────────────────────────────────────────────

def parse_legal_document(file_path: Path, llm=None) -> tuple[Document, dict]:
    """
    Parse a PDF into a LlamaIndex Document with clean string metadata.

    PDF extraction: PyMuPDF fast-path for born-digital, Docling fallback for scans.
    Metadata: regex fast-path, LLM fallback when regex fails (pass llm=None to skip).
    All metadata fields guaranteed strings — never None or match objects.
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
                logger.info(f"PyMuPDF fast path: {file_path.name}")
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {file_path.name}: {e}")
            md_text = None

    if md_text is None:
        logger.info(f"Docling fallback: {file_path.name}")
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        md_text = result.document.export_to_markdown()

    meta = _extract_metadata_regex(md_text)

    if (meta["law_nr"] is None or meta["effective_date"] is None) and llm is not None:
        logger.info(f"Regex metadata failed for {file_path.name} — using LLM fallback")
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


# ── JSON support (legislatie.just.ro format) ──────────────────────────────────

# Noise patterns injected by the portal's browser UI into text_complet.
# These appear at the tail of the field and carry no legal content.
_JSON_NOISE_PATTERNS = [
    r'×\s*Inchide\s+Versiune Beta.*$',           # beta banner
    r'×\s*Inchide\s+Datorita faptului.*$',        # browser warning
    r'Reveniti in topul paginii.*$',               # navigation link
    r'Forma printabilă\s*$',                       # print link
    r'Ce browser sa aleg\?.*$',                    # browser upgrade prompt
]
_JSON_NOISE_RE = re.compile(
    '|'.join(_JSON_NOISE_PATTERNS),
    re.IGNORECASE | re.DOTALL
)


def _clean_json_text(raw_text: str) -> str:
    """
    Remove browser UI noise from legislatie.just.ro text_complet field.
    Noise always appears in the tail — we strip from the first match onward.
    """
    cleaned = _JSON_NOISE_RE.sub('', raw_text)
    return cleaned.strip()


def parse_json_document(file_path: Path) -> tuple[Document, dict]:
    """
    Parse a legislatie.just.ro JSON file into a LlamaIndex Document.

    Metadata is read directly from JSON fields — no regex or LLM fallback needed.
    Structured fields available: tip_act, numar, data_vigoare, emitent,
    publicatie, titlu, in_vigoare, link, id.

    law_nr is constructed as "{tip_act} {numar}" when numar is non-zero,
    or falls back to the id field.

    text_complet is used as the document body; text_snippet is ignored
    (it is a subset of text_complet with no additional content).

    All metadata fields guaranteed strings.
    """
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse JSON {file_path.name}: {e}")
        raise

    # ── Metadata — read directly from structured fields ───────────────────────
    tip_act = str(raw.get("tip_act", "")).strip()
    numar   = str(raw.get("numar", "0")).strip()
    doc_id  = str(raw.get("id", file_path.stem)).strip()

    # Build law_nr: prefer "TIP_ACT numar" when numar is meaningful
    if numar and numar != "0":
        law_nr = f"{tip_act} {numar}".strip()
    else:
        # Some acts (STATUT, etc.) have numar=0; use id as disambiguator
        law_nr = f"{tip_act} [{doc_id}]".strip() if tip_act else doc_id

    # data_vigoare is ISO YYYY-MM-DD — convert to DD.MM.YYYY for consistency
    data_vigoare_raw = str(raw.get("data_vigoare", "")).strip()
    if re.match(r'\d{4}-\d{2}-\d{2}', data_vigoare_raw):
        parts = data_vigoare_raw.split("-")
        effective_date = f"{parts[2]}.{parts[1]}.{parts[0]}"
    else:
        effective_date = data_vigoare_raw or "unknown"

    metadata = {
        "filename":       file_path.stem,
        "law_nr":         law_nr or "unknown",
        "effective_date": effective_date,
        "source_type":    "romanian_legal",
        # JSON-exclusive fields — kept for manifest and future filtering
        "tip_act":        tip_act,
        "emitent":        str(raw.get("emitent", "")).strip(),
        "in_vigoare":     str(raw.get("in_vigoare", "true")),
        "source_url":     str(raw.get("link", "")).strip(),
        "doc_id":         doc_id,
    }

    # ── Text — clean text_complet ─────────────────────────────────────────────
    raw_text = raw.get("text_complet", "") or raw.get("text_snippet", "")
    if not raw_text.strip():
        logger.warning(f"Empty text_complet in {file_path.name} — document will be low-quality")
        raw_text = raw.get("titlu", "")

    clean_text = _clean_json_text(raw_text)

    # Prepend title as context header (equivalent to PDF preamble)
    titlu = str(raw.get("titlu", "")).strip()
    if titlu and not clean_text.startswith(titlu[:50]):
        clean_text = f"{titlu}\n\n{clean_text}"

    return Document(text=clean_text, metadata=metadata), metadata
````

---

## `src/pipeline.py` ← CHANGED in v1.5

Two targeted changes only — imports and file collection/routing. Everything else is identical to v1.4.

```python
import gc
import json
import logging
from pathlib import Path
from tqdm import tqdm
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from src.config import (
    RAW_DIR, ARTIFACTS_DIR, EMBED_MODEL,
    OLLAMA_MODEL_RO, OLLAMA_MODEL_LEGAL
)
# v1.5: import parse_json_document alongside existing imports
from src.parsers import parse_legal_document, parse_json_document, node_parser, LegalTextSplitter
from src.extractors import extract_rules, create_contextual_summary
from src.graph_builder import build_legal_graph

logger = logging.getLogger(__name__)


def load_manifest() -> dict:
    p = ARTIFACTS_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_manifest(manifest: dict):
    (ARTIFACTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_end_to_end(
    limit: int = 100,
    resume: bool = True,
    skip_kg: bool = False,
    skip_vector: bool = False,
    seed_glossary: bool = False,
    quick: bool = False,
):
    """
    quick=True  → vector + summaries only (fastest smoke test)
    resume=True → checkpoints active (default); --no-resume for full rebuild
    Accepts both .pdf and .json files in data/raw/.
    """
    if quick:
        skip_kg = True
        logger.info("--quick mode: skipping KG, rule extraction, glossary seeding")

    for subdir in ["summaries", "rules", "graph"]:
        (ARTIFACTS_DIR / subdir).mkdir(exist_ok=True, parents=True)

    print("🤖 Loading RoLlama3.1 (summaries + metadata fallback)...")
    llm_ro        = Ollama(model=OLLAMA_MODEL_RO, temperature=0.1)
    llm_legal     = None   # lazy-loaded on first article chunk (v1.3 patch 1)
    embed_model   = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    legal_splitter = LegalTextSplitter()
    manifest      = load_manifest()

    # ── v1.5: Collect both PDF and JSON files, sorted, respect limit ──────────
    pdf_files  = sorted(RAW_DIR.glob("*.pdf"))
    json_files = sorted(RAW_DIR.glob("*.json"))
    files      = (pdf_files + json_files)[:limit]
    # ─────────────────────────────────────────────────────────────────────────

    if not files:
        print(f"⚠️  No PDF or JSON files found in {RAW_DIR} — add files and re-run")
        return None, None

    n_pdf  = sum(1 for f in files if f.suffix == ".pdf")
    n_json = sum(1 for f in files if f.suffix == ".json")
    print(f"📄 Found {len(files)} files ({n_pdf} PDF, {n_json} JSON) "
          f"(limit={limit}, quick={quick}, resume={resume})")

    all_nodes = []

    # ── Per-document loop ─────────────────────────────────────────────────────
    for f in tqdm(files, desc="📄 Processing"):

        # ── v1.5: Route to correct parser by file extension ───────────────────
        if f.suffix.lower() == ".json":
            doc, raw_meta = parse_json_document(f)
            # JSON has structured metadata — no LLM fallback needed
        else:
            doc, raw_meta = parse_legal_document(f, llm=llm_ro)
        # ─────────────────────────────────────────────────────────────────────

        # Two-stage chunking (v1.4): legal markers → hierarchical sub-chunking
        pre_chunks = legal_splitter.split_text(doc.text)
        sub_docs   = [
            Document(text=chunk, metadata=doc.metadata)
            for chunk in pre_chunks
        ]
        nodes = node_parser.get_nodes_from_documents(sub_docs)

        for node in nodes:
            ctx_prefix = f"[{node.metadata.get('law_nr')} | {node.metadata.get('filename')}]\n"
            node.text = ctx_prefix + node.text
            node.metadata["context_prefix"] = ctx_prefix

            is_article_chunk = any(
                kw in node.text.upper() for kw in ["ARTICOL", "ART.", "ALIN."]
            )

            summary_path = ARTIFACTS_DIR / "summaries" / f"{node.node_id}.json"
            rules_path   = ARTIFACTS_DIR / "rules"     / f"{node.node_id}.json"
            already_done = resume and summary_path.exists() and rules_path.exists()

            if not already_done and is_article_chunk:
                summary = create_contextual_summary(llm_ro, node.text, node.metadata)
                summary_path.write_text(
                    json.dumps({
                        "summary":        summary,
                        "law_nr":         node.metadata.get("law_nr"),
                        "filename":       node.metadata.get("filename"),
                        "effective_date": node.metadata.get("effective_date"),
                    }, ensure_ascii=False),
                    encoding="utf-8"
                )

                if not quick:
                    if llm_legal is None:
                        print("\n🤖 Loading LawMate (rule extraction)...")
                        llm_legal = Ollama(model=OLLAMA_MODEL_LEGAL, temperature=0.0)
                    rules = extract_rules(llm_legal, node.text, node.node_id)
                    rules_path.write_text(
                        json.dumps([r.model_dump() for r in rules], ensure_ascii=False),
                        encoding="utf-8"
                    )
                else:
                    rules_path.write_text("[]", encoding="utf-8")

            manifest[node.node_id] = {
                "law_nr":         node.metadata.get("law_nr"),
                "filename":       node.metadata.get("filename"),
                "effective_date": node.metadata.get("effective_date"),
                "source_type":    node.metadata.get("source_type", "romanian_legal"),
                "file_format":    f.suffix.lstrip("."),   # "pdf" or "json"
                "is_article_chunk": is_article_chunk,
                "has_summary":    is_article_chunk,
                "has_rules":      is_article_chunk and not quick,
            }
            all_nodes.append(node)

        save_manifest(manifest)

    if llm_legal is not None:
        print("🧹 Releasing LawMate from memory...")
        del llm_legal
        gc.collect()

    print(f"✅ Processed {len(files)} files → {len(all_nodes)} nodes "
          f"({n_pdf} PDF + {n_json} JSON)")

    # ── Vector Index ──────────────────────────────────────────────────────────
    vector_index = None
    if not skip_vector:
        vector_dir    = ARTIFACTS_DIR / "vector_index"
        chroma_client = chromadb.PersistentClient(path=str(vector_dir))
        collection    = chroma_client.get_or_create_collection("legal_corpus")

        if resume and collection.count() > 0:
            print(f"⏭️  Vector index exists ({collection.count()} vectors) — skipping")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store, embed_model=embed_model
            )
        else:
            print("🔢 Building vector index...")
            vector_store    = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index    = VectorStoreIndex(
                nodes=all_nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            print(f"✅ Vector index: {collection.count()} vectors")

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    graph_index = None
    if not skip_kg:
        print("🤖 Loading LawMate for KG extraction...")
        llm_legal_kg = Ollama(model=OLLAMA_MODEL_LEGAL, temperature=0.0)
        graph_index  = build_legal_graph(
            all_nodes,
            llm_legal=llm_legal_kg,
            seed_glossary=seed_glossary
        )
        del llm_legal_kg
        gc.collect()

    print(f"\n🎉 Pipeline complete!")
    print(f"   Files: {len(files)} ({n_pdf} PDF + {n_json} JSON) | Nodes: {len(all_nodes)}")
    print(f"   Artefacts: {ARTIFACTS_DIR}")
    return vector_index, graph_index
```

---

## All other files unchanged from v1.4

`config.py`, `extractors.py`, `graph_builder.py`, `query_harness.py`, `run_end_to_end.py`, `requirements.txt`, `setup_env.sh` — identical to v1.4.

---

## JSON field mapping decision log

| JSON field                                | Maps to                      | Notes                                                  |
| ----------------------------------------- | ---------------------------- | ------------------------------------------------------ |
| `text_complet`                            | `Document.text`              | Primary body — cleaned of browser noise                |
| `text_snippet`                            | ignored                      | Strict subset of `text_complet`                        |
| `titlu`                                   | prepended to text as header  | Equivalent to PDF preamble                             |
| `tip_act` + `numar`                       | `metadata["law_nr"]`         | `"STATUT [109831]"` when numar=0                       |
| `data_vigoare` (ISO)                      | `metadata["effective_date"]` | Converted to DD.MM.YYYY for consistency with PDF path  |
| `emitent`                                 | `metadata["emitent"]`        | JSON-exclusive — preserved for future filtering        |
| `in_vigoare`                              | `metadata["in_vigoare"]`     | String `"true"/"false"` — temporal filtering in v2     |
| `link`                                    | `metadata["source_url"]`     | Provenance — useful for harness citations              |
| `id`                                      | `metadata["doc_id"]`         | Portal identifier — used as disambiguator when numar=0 |
| `html_size`, `descarcat_la`, `publicatie` | dropped                      | No value to pipeline or retrieval                      |

---

## Noise cleaning decision

`text_complet` from the portal has a fixed tail pattern: browser version warning, navigation links, and browser upgrade prompts. These appear after the last legal article. The `_JSON_NOISE_RE` pattern anchors to the first noise marker and strips everything from there to end-of-string. If the patterns evolve across portal versions, only `_JSON_NOISE_PATTERNS` needs updating — isolated in one place.

---
