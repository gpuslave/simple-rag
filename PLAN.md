# Python RAG CLI Architecture

## Summary

Build a greenfield, CLI-only modular monolith for a single synchronized PDF corpus of up to 10,000 pages.

```text
PDF directory → extraction → page chunks → embeddings → local Qdrant
Question → dense retrieval → evidence prompt → generation model → validated citations
```

Use Python 3.12, Nix flake, uv, LangChain, PyMuPDF, local on-disk Qdrant, Typer, and Pydantic. RouterAI is the configurable Model Gateway for embeddings and generation.

## Architecture and Interfaces

- Keep project-owned domain/application interfaces around LangChain:
  - `PdfExtractor`
  - `EmbeddingProvider`
  - `Retriever`
  - `AnswerGenerator`
  - `CorpusSynchronizer`
- Implement a dense Qdrant retriever initially; preserve the `Retriever` boundary for later BM25, hybrid, ensemble, and chat-aware retrieval.
- Define domain types for `Document`, `Page`, `Chunk`, `RetrievedChunk`, `Claim`, `Citation`, `AnswerResult`, `SyncReport`, and `IndexFingerprint`.
- Use these CLI commands:
  - `rag doctor` — validate configuration, credentials, model capabilities, embedding dimensions, and index compatibility.
  - `rag sync [--dry-run] [--json]` — reconcile the configured directory.
  - `rag ask QUESTION [--json]` — answer one independent question.
  - `rag status` — show corpus/index statistics and last synchronization state.
  - `rag eval DATASET` — run the local gold-set report.
  - `rag rebuild-index --yes` — explicitly recreate an incompatible index.

## Configuration and Storage

- Store non-secret settings in `rag.toml`; require explicit generation and embedding model slugs.
- Default RouterAI base URL to `https://routerai.ru/api/v1`; read `ROUTERAI_API_KEY` only from the environment.
- Support optional RouterAI generation-provider routing settings; leave routing unrestricted when absent.
- Require the generation model to advertise strict structured-output support. `z-ai/glm-5.3-flash` is a compatible example. [RouterAI capabilities](https://routerai.ru/models/z-ai/glm-5.3-flash)
- Persist Qdrant locally without a separate service. [Qdrant local mode](https://qdrant.tech/documentation/frameworks/langchain/)
- Maintain an atomic local state manifest containing:
  - Schema version
  - Canonical corpus root
  - Embedding slug and discovered vector dimension
  - Chunking configuration
  - Collection name
  - Last successful sync summary
- Fail on fingerprint mismatch and require an explicit rebuild when the corpus root, embedding model, vector dimension, schema, or chunking settings change.

## Ingestion and Retrieval

- Recursively discover case-insensitive `.pdf` files without following symlinks.
- Identify documents by canonical source path and detect changes with SHA-256.
- Extract text with PyMuPDF in page order; do not OCR.
- Warn and skip textless pages while indexing the remaining document pages.
- Split each page independently using LangChain’s recursive splitter:
  - 800 estimated tokens per chunk
  - 100-token overlap
  - Never cross page boundaries
- Store deterministic point IDs and payload metadata: document ID, path, filename, content hash, viewer page, optional PDF page label, chunk index, and text.
- For changed documents, fully extract/embed the new version before replacing old points. Clean partial new points after failure and preserve the old version.
- Update successful documents during partial sync failures, but postpone every missing-file deletion until a run completes without document failures.
- Skip unchanged documents. Treat a rename as removal plus addition.
- Dense retrieval defaults:
  - Fetch 30 nearest candidates.
  - Remove duplicates.
  - Limit repeated evidence to three chunks per page.
  - Supply up to 15 chunks to generation.
  - Do not impose an uncalibrated similarity threshold.

## Answer and Citation Contract

- Generate answers in Russian regardless of source/question language.
- Prompt the generation model to use retrieved evidence exclusively and return `insufficient_evidence` when unsupported.
- Require strict structured output:

```json
{
  "status": "answered",
  "claims": [
    {
      "text": "Фактическое утверждение.",
      "source_ids": ["S1", "S3"]
    }
  ]
}
```

- An insufficient response has `status: "insufficient_evidence"` and no claims; the application renders a fixed Russian refusal.
- Reject malformed output, unknown source IDs, uncited factual claims, or models without structured-output support. Never print an unvalidated answer.
- Render human citations as `[filename, PDF p. N, label X]`; omit the label when absent.
- `--json` returns status, claims, citations, paths, pages, labels, similarity scores, excerpts, and configured model slugs.

## Documentation and Decisions

- Create `CONTEXT.md` defining Corpus, Document, Page, Chunk, Retriever, Vector Database, Embedding Model, Generation Model, Model Gateway, Claim, and Citation.
- Record focused ADRs for:
  - Modular monolith with LangChain behind project-owned boundaries.
  - Local Qdrant and dense-only v1 retrieval.
  - Strict claim-level citation validation and model capability requirements.
- Document setup, `rag.toml`, CLI workflows, index rebuilds, RouterAI data transmission, and future extension points in the README.

## Test and Evaluation Plan

- Unit-test page-bounded splitting, stable identities, fingerprints, citation validation, rendering, and refusal behavior.
- Integration-test local Qdrant sync with fake embeddings:
  - Initial and idempotent sync
  - Changed document replacement
  - Rename/remove behavior
  - Deferred deletion after failures
  - Partial-upload cleanup
  - Textless-page warnings
- Test retrieval ranking, per-page caps, bilingual evidence, strict structured responses, human output, and JSON output.
- Keep RouterAI tests opt-in and credentialed: capability discovery, embedding dimension probing, strict generation, and a complete live question.
- Define a JSONL gold-set format covering expected pages, answerable/unanswerable questions, and bilingual queries. `rag eval` reports recall@15, MRR, refusal results, citation validity, and language compliance without CI gating.

## Assumptions

- No HTTP API, authentication, chat sessions, OCR, reranking, BM25/hybrid retrieval, or CI in v1.
- PDFs and retrieved excerpts may be transmitted to RouterAI.
- Printed page labels are used only when present in PDF metadata; viewer page numbers remain authoritative.
- Concurrent CLI access is unsupported; commands use a local lock and fail fast when another operation owns the index.
