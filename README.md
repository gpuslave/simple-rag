# Page-cited RAG CLI

[![CI](https://github.com/gpuslave/simple-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gpuslave/simple-rag/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python 3.12 CLI for a local PDF corpus, RouterAI models, and page-level citations.

## Development

```bash
nix develop
just setup
just test
```

Run `just` to list every available command:

```text
just setup                 # install locked dependencies
just test                  # run offline tests
just test-live             # run opt-in RouterAI integration tests
just coverage              # run offline tests with an 85% coverage floor
just lint                  # check Python lint rules and formatting
just format                # apply safe lint fixes and formatting
just typecheck             # run strict static type checks
just doctor                # run live readiness checks
just doctor rag.toml       # use another configuration file
just sync                  # synchronize the configured PDF corpus
just sync rag.toml --dry-run # preview additions, updates, and removals
just sync rag.toml --json  # emit a machine-readable sync report
just ask "What is this document about?" rag.toml # answer with page citations
just ask "What is this document about?" rag.toml --json # emit validated JSON
just check                 # run every offline CI check
just build                 # build wheel and source package
```

The justfile automatically loads the ignored `.env.example` when present.

Copy `rag.toml.example` to `rag.toml`, edit the three required values, then provide the secret only through the environment:

```bash
export ROUTERAI_API_KEY='...'
just doctor rag.toml
```

`doctor` checks that the corpus exists, local state is writable, an eligible generation endpoint advertises strict structured outputs, and the embedding model returns a non-empty numeric vector. The embedding probe is a live API request.

Generation behavior is explicit and required:

```toml
[generation]
reasoning_effort = "low"
temperature = 0.2
max_tokens = 8192
```

`reasoning_effort` accepts `low`, `high`, or `max`; temperature accepts values from 0 through 2; and `max_tokens` must be positive. We use low reasoning and conservative sampling for economical, grounded answers. Top-p, seed, penalties, and reasoning output remain unset. Provider routing under `[generation.provider]` is optional; omitting it leaves RouterAI routing unrestricted.

`sync` recursively treats the configured corpus as authoritative. It adds and updates PDFs before removing older versions, skips unchanged content without embedding it, and removes missing PDFs only after every present PDF succeeds. Use `--dry-run` to inspect the same human or JSON change report without modifying Qdrant.

`ask` embeds one independent question, retrieves up to 30 dense candidates, keeps at most three chunks per page and 15 chunks overall, and asks the Generation Model for a strict structured Russian answer. Every printed factual claim has validated PDF-page citations; unsupported questions receive a fixed insufficient-evidence response. Configure these limits under `[retrieval]`.

Architecture terminology is in [CONTEXT.md](CONTEXT.md); decisions are in [docs/adr](docs/adr).
