import gc
import json
import logging
from pathlib import Path

import chromadb
from tqdm import tqdm

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import ARTIFACTS_DIR, EMBED_MODEL, OLLAMA_MODEL_LEGAL, OLLAMA_MODEL_RO, RAW_DIR
from src.extractors import create_contextual_summary, extract_rules
from src.graph_builder import build_legal_graph
from src.parsers import LegalTextSplitter, node_parser, parse_json_document, parse_legal_document

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
    llm_ro = Ollama(model=OLLAMA_MODEL_RO, temperature=0.1)
    llm_legal = None
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    legal_splitter = LegalTextSplitter()
    manifest = load_manifest()

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    json_files = sorted(RAW_DIR.glob("*.json"))
    files = sorted(pdf_files + json_files)[:limit]

    n_pdf = sum(1 for f in files if f.suffix.lower() == ".pdf")
    n_json = sum(1 for f in files if f.suffix.lower() == ".json")
    all_nodes = []

    for f in tqdm(files, desc="Parsing files"):
        if f.suffix.lower() == ".json":
            doc, _ = parse_json_document(f)
        else:
            doc, _ = parse_legal_document(f, llm=llm_ro)

        pre_chunks = legal_splitter.split_text(doc.text)
        sub_docs = [Document(text=chunk, metadata=doc.metadata) for chunk in pre_chunks]
        nodes = node_parser.get_nodes_from_documents(sub_docs)

        for node in nodes:
            ctx_prefix = f"[{node.metadata.get('law_nr')} | {node.metadata.get('filename')}]\n"
            node.text = ctx_prefix + node.text
            node.metadata["context_prefix"] = ctx_prefix

            is_article_chunk = any(kw in node.text.upper() for kw in ["ARTICOL", "ART.", "ALIN."])

            summary_path = ARTIFACTS_DIR / "summaries" / f"{node.node_id}.json"
            rules_path = ARTIFACTS_DIR / "rules" / f"{node.node_id}.json"
            already_done = resume and summary_path.exists() and rules_path.exists()

            if not already_done and is_article_chunk:
                summary = create_contextual_summary(llm_ro, node.text, node.metadata)
                summary_path.write_text(
                    json.dumps(
                        {
                            "summary": summary,
                            "law_nr": node.metadata.get("law_nr"),
                            "filename": node.metadata.get("filename"),
                            "effective_date": node.metadata.get("effective_date"),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                if not quick:
                    if llm_legal is None:
                        print("\n🤖 Loading LawMate (rule extraction)...")
                        llm_legal = Ollama(model=OLLAMA_MODEL_LEGAL, temperature=0.0)
                    rules = extract_rules(llm_legal, node.text, node.node_id)
                    rules_path.write_text(
                        json.dumps([r.model_dump() for r in rules], ensure_ascii=False),
                        encoding="utf-8",
                    )
                else:
                    rules_path.write_text("[]", encoding="utf-8")

            manifest[node.node_id] = {
                "law_nr": node.metadata.get("law_nr"),
                "filename": node.metadata.get("filename"),
                "effective_date": node.metadata.get("effective_date"),
                "source_type": node.metadata.get("source_type", "romanian_legal"),
                "file_format": f.suffix.lstrip("."),
                "is_article_chunk": is_article_chunk,
                "has_summary": is_article_chunk,
                "has_rules": is_article_chunk and not quick,
            }
            all_nodes.append(node)

        save_manifest(manifest)

    if llm_legal is not None:
        print("🧹 Releasing LawMate from memory...")
        del llm_legal
        gc.collect()

    print(f"✅ Processed {len(files)} files → {len(all_nodes)} nodes ({n_pdf} PDF + {n_json} JSON)")

    vector_index = None
    if not skip_vector:
        vector_dir = ARTIFACTS_DIR / "vector_index"
        chroma_client = chromadb.PersistentClient(path=str(vector_dir))
        collection = chroma_client.get_or_create_collection("legal_corpus")

        if resume and collection.count() > 0:
            print(f"⏭️  Vector index exists ({collection.count()} vectors) — skipping")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        else:
            print("🔢 Building vector index...")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex(
                nodes=all_nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            print(f"✅ Vector index: {collection.count()} vectors")

    graph_index = None
    if not skip_kg:
        print("🤖 Loading LawMate for KG extraction...")
        llm_legal_kg = Ollama(model=OLLAMA_MODEL_LEGAL, temperature=0.0)
        graph_index = build_legal_graph(all_nodes, llm_legal=llm_legal_kg, seed_glossary=seed_glossary)
        del llm_legal_kg
        gc.collect()

    print("\n🎉 Pipeline complete!")
    print(f"   Files: {len(files)} ({n_pdf} PDF + {n_json} JSON) | Nodes: {len(all_nodes)}")
    print(f"   Artefacts: {ARTIFACTS_DIR}")
    return vector_index, graph_index
