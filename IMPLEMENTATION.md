# RAG CLI Implementation Tickets

Source architecture: [PLAN.md](PLAN.md)

## Dependency frontier

1. Start with ticket 01.
2. Ticket 02 starts after ticket 01.
3. Tickets 03 and 04 can run in parallel after ticket 02.
4. Ticket 05 starts after ticket 03.
5. Ticket 06 starts after tickets 03 and 04.
6. Ticket 07 starts after tickets 05 and 06.

# 01: Establish a runnable RAG CLI and configuration contract

**What to build:** Create the runnable Python application foundation so an operator can enter the Nix development environment, install locked dependencies, invoke the `rag` command, load validated settings, and use `rag doctor` to verify that the system is ready before indexing documents. Keep LangChain and infrastructure details behind project-owned domain and application interfaces so future retrievers do not change the CLI contract.

**Blocked by:** None (can start immediately).

**Status:** completed

- [x] Python 3.12, the Nix development environment, uv dependency locking, packaging, and the `rag` console entry point work from a clean checkout.
- [x] Configuration requires corpus, generation-model, and embedding-model settings; defaults the Model Gateway URL to RouterAI; and reads the API key only from `ROUTERAI_API_KEY`.
- [x] Optional RouterAI generation-provider routing is supported and remains unrestricted when omitted.
- [x] Invalid or missing configuration produces concise actionable errors without exposing secrets.
- [x] `rag doctor` verifies the corpus directory, writable local state location, RouterAI connectivity, generation-model strict structured-output support, and embedding vector dimensionality.
- [x] A generation model that does not advertise strict structured outputs is rejected before querying is possible.
- [x] The core ports and domain types exist without exposing LangChain types through their public contracts.
- [x] The domain glossary defines Corpus, Document, Page, Chunk, Retriever, Vector Database, Embedding Model, Generation Model, Model Gateway, Claim, and Citation.
- [x] ADRs record the modular-monolith/LangChain boundary, local Qdrant dense retrieval, and validated claim-level citation decisions.
- [x] Automated tests cover valid configuration, missing secrets, invalid models, capability failures, and successful mocked diagnostics.

# 02: Index the first PDF into local Qdrant

**What to build:** Make `rag sync` complete a narrow ingestion path from a configured directory containing one text PDF through page extraction, page-bounded chunking, RouterAI embeddings, and persistent local Qdrant storage. The resulting points must contain everything later retrieval needs to produce page-level citations.

**Blocked by:** 01: Establish a runnable RAG CLI and configuration contract.

**Status:** completed

- [x] `rag sync` discovers a PDF in the configured corpus directory and extracts text in page reading order with PyMuPDF.
- [x] Textless pages are skipped with visible warnings and are listed in the synchronization report.
- [x] Every chunk remains within one PDF page and uses the configured 800-token size and 100-token overlap defaults.
- [x] Document, page, chunk, and point identities are deterministic across repeated processing of unchanged content.
- [x] RouterAI produces document embeddings through the configured Embedding Model, with raw strings sent through the compatible LangChain adapter.
- [x] The first successful embedding determines the vector dimension used to create the persistent local Qdrant collection.
- [x] Stored payloads include canonical document identity, source path, filename, content hash, viewer page number, optional PDF page label, chunk index, and chunk text.
- [x] Human and `--json` synchronization reports show indexed documents, pages, chunks, skipped pages, and failures.
- [x] A second run against the unchanged single document does not create duplicate points.
- [x] Tests exercise extraction, page labels, textless pages, chunk boundaries, deterministic IDs, embedding failures, and persistent retrieval from local Qdrant.

# 03: Safely synchronize the authoritative PDF directory

**What to build:** Extend the initial ingestion path into authoritative recursive directory synchronization. Operators must be able to preview and apply additions, updates, renames, and removals without losing the last usable version of a document when extraction, embedding, or upload fails.

**Blocked by:** 02: Index the first PDF into local Qdrant.

**Status:** ready-for-agent

- [ ] Discovery recursively includes case-insensitive PDF extensions and does not follow symlinks.
- [ ] Canonical paths define document identity, while SHA-256 detects unchanged and changed content.
- [ ] Unchanged documents are skipped without embedding calls.
- [ ] A changed document is fully extracted and embedded before its previous indexed version is removed.
- [ ] Failed uploads clean up partial points for the new version and leave the previous usable version intact.
- [ ] Successfully processed documents may advance during a partially failed run, but no missing-file deletion occurs unless every present document succeeds.
- [ ] A successful failure-free run removes documents no longer present in the authoritative directory.
- [ ] Renaming a PDF is reported and applied as one removal and one addition.
- [ ] `--dry-run` reports the exact intended additions, updates, skips, warnings, and removals without changing Qdrant or synchronization state.
- [ ] Repeating a successful synchronization is idempotent.
- [ ] Human and JSON reports distinguish completed work, deferred removals, warnings, and actionable failures.
- [ ] Integration tests cover additions, changes, unchanged files, renames, removals, extraction failure, embedding failure, partial upload cleanup, and deferred deletion.

# 04: Answer one question with validated page citations

**What to build:** Make `rag ask` answer one independent question from indexed evidence. The complete path must embed the question, retrieve relevant chunks, request a strict structured Russian answer from the configured Generation Model, validate every claim's source references, and render trustworthy page-level citations.

**Blocked by:** 02: Index the first PDF into local Qdrant.

**Status:** ready-for-agent

- [ ] The question is embedded with the same configured Embedding Model and collection contract used for ingestion.
- [ ] Dense retrieval fetches 30 candidates, removes duplicate evidence, limits evidence to three chunks per page, and supplies at most 15 chunks to generation.
- [ ] Retrieval applies no arbitrary similarity threshold.
- [ ] Retrieved chunks receive stable request-local source IDs and carry filename, viewer page, optional page label, score, and text into answer generation.
- [ ] The Generation Model receives only the question, instructions, and retrieved evidence and is instructed to answer in Russian without outside knowledge.
- [ ] Strict output represents either an answered result containing factual claims with source IDs or an `insufficient_evidence` result with no claims.
- [ ] Validation rejects malformed output, empty citations on factual claims, duplicate source IDs, and source IDs not present in retrieved evidence.
- [ ] Invalid model output fails without printing an unvalidated answer.
- [ ] Insufficient evidence renders a fixed Russian refusal.
- [ ] Human output attaches `[filename, PDF p. N, label X]` citations to each factual claim and omits unavailable labels.
- [ ] `--json` includes status, claims, citations, paths, pages, labels, scores, excerpts, and the configured model slugs.
- [ ] Tests cover Russian and English evidence, ranking and per-page limits, valid multi-source claims, refusal, malformed output, invented sources, human rendering, and JSON output.

# 05: Inspect and rebuild incompatible indexes

**What to build:** Give operators a safe lifecycle for local index state. They must be able to inspect the corpus/index, understand why configuration is incompatible, avoid concurrent corruption, and explicitly rebuild after changing the corpus root, embedding contract, schema, or chunking policy.

**Blocked by:** 03: Safely synchronize the authoritative PDF directory.

**Status:** ready-for-agent

- [ ] Atomic local state records the schema version, canonical corpus root, collection name, embedding slug, vector dimension, chunking settings, and last successful synchronization summary.
- [ ] `rag status` reports document, page, chunk, warning, model, fingerprint, storage, and last-sync information without contacting the Generation Model.
- [ ] `rag doctor`, `rag sync`, and `rag ask` fail with an actionable incompatibility explanation when the stored fingerprint differs from configuration.
- [ ] Generation-model and provider-routing changes do not require reindexing when the embedding/index fingerprint is otherwise unchanged.
- [ ] Corpus-root, embedding-model, vector-dimension, schema-version, or chunking changes require an explicit rebuild.
- [ ] `rag rebuild-index` refuses destructive work without `--yes`.
- [ ] A confirmed rebuild recreates the collection and synchronization state from the authoritative source directory using the current configuration.
- [ ] A local operation lock makes overlapping commands fail fast with a clear message rather than opening the same local Qdrant storage concurrently.
- [ ] State writes are atomic and interrupted writes do not leave a partially serialized manifest.
- [ ] Tests cover every compatible and incompatible fingerprint field, confirmation behavior, rebuild results, interrupted state writes, and lock contention.

# 06: Evaluate retrieval and answer quality locally

**What to build:** Provide a repeatable local evaluation workflow that measures whether retrieval finds the expected PDF pages and whether answer generation follows the refusal, language, and citation contracts. Keep normal tests offline and deterministic while allowing explicitly requested live RouterAI verification.

**Blocked by:** 03: Safely synchronize the authoritative PDF directory; 04: Answer one question with validated page citations.

**Status:** ready-for-agent

- [ ] A documented JSONL dataset format represents bilingual questions, expected documents/pages, answerability, and optional expected answer facts.
- [ ] `rag eval DATASET` runs against the configured local corpus and produces a human-readable report plus optional JSON output.
- [ ] The report includes page recall@15, mean reciprocal rank, answer/refusal outcomes, citation validity, and Russian-language compliance.
- [ ] Evaluation reports results without enforcing CI thresholds or requiring a CI system.
- [ ] Offline tests use deterministic fake embeddings and model responses to cover the full ingestion, retrieval, generation, validation, and rendering path.
- [ ] PDF fixtures cover multiple pages, labels, textless pages, Russian text, English text, changed content, and deleted content.
- [ ] Credentialed live tests are opt-in and excluded from the default test command.
- [ ] Live checks cover RouterAI capability discovery, embedding dimension probing, document/query embeddings, strict structured generation, and one complete cited question.
- [ ] Live failures clearly distinguish credentials, model compatibility, rate limiting, gateway failure, and invalid output.

# 07: Publish the operator and extension guide

**What to build:** Make the completed CLI usable by a new operator and understandable to a future maintainer. Documentation must describe the working system rather than aspirations and must show how the existing boundaries admit future retrievers and conversational querying without promising those features in v1.

**Blocked by:** 05: Inspect and rebuild incompatible indexes; 06: Evaluate retrieval and answer quality locally.

**Status:** ready-for-agent

- [ ] Setup instructions cover Nix, uv, RouterAI credentials, configuration, and first-run diagnostics from a clean checkout.
- [ ] Example workflows cover doctor, synchronization and dry runs, asking questions, JSON output, status, evaluation, and confirmed rebuilding.
- [ ] Configuration documentation distinguishes required model slugs, non-secret settings, secrets, provider routing, chunking, retrieval limits, and storage paths.
- [ ] The guide explains that source chunks and questions are transmitted to RouterAI and that local Qdrant persists extracted text and vectors.
- [ ] Citation documentation distinguishes viewer page numbers from optional PDF metadata labels and explains textless-page warnings.
- [ ] Troubleshooting covers unsupported structured output, embedding/index mismatch, malformed model output, failed synchronization, deferred removals, lock contention, and missing citations.
- [ ] Architecture documentation shows LangChain contained behind project-owned interfaces and identifies the extension points for BM25, hybrid, ensemble, reranking, HTTP, and chat sessions.
- [ ] The glossary and ADRs agree with the implemented terminology and behavior.
- [ ] The documented clean-room workflow successfully configures, indexes, queries, evaluates, inspects, and rebuilds the application locally.
