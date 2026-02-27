# explore-law-rag (POC v1.5 scaffold)

This repository now includes a POC implementation aligned with `agent.md` v1.5:
- JSON support via `parse_json_document`
- JSON text noise cleanup via `_clean_json_text`
- Unified PDF + JSON routing in the pipeline
- Manifest enrichment with file format metadata

## Important limitation
Running experiments and installing model runtimes (e.g., Ollama) are intentionally out of scope for this task.

## What is implemented in this environment
- Source-level implementation for parsers and pipeline orchestration.
- Minimal stubs for extractors and graph build so the code structure is complete.

## Not executed here
- End-to-end model runs
- Ollama model pulls/inference
- Full vector/KG build against real corpora
