# Domain Glossary

- **Corpus:** The single authoritative directory tree of PDF source files synchronized into the index.
- **Document:** One PDF identified by its canonical source path and versioned by its content hash.
- **Page:** A viewer-numbered PDF page. An optional PDF metadata label may also be present.
- **Chunk:** A page-bounded text segment embedded and stored as one retrievable unit.
- **Retriever:** An application port that ranks stored chunks for a question without exposing storage-library types.
- **Vector Database:** The local Qdrant store containing chunk vectors and citation metadata.
- **Embedding Model:** The configured RouterAI model that maps document chunks and questions into vectors.
- **Generation Model:** The configured RouterAI model that turns retrieved evidence into a structured Russian answer.
- **Model Gateway:** The OpenAI-compatible RouterAI API used to reach embedding and generation models.
- **Claim:** One factual statement in an answer, linked to one or more request-local source IDs.
- **Citation:** Validated source metadata that resolves a source ID to a PDF path, viewer page, optional label, score, and excerpt.

