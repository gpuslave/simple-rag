# Page-cited RAG CLI

[![CI](https://github.com/gpuslave/simple-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gpuslave/simple-rag/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/github/gpuslave/simple-rag/graph/badge.svg?branch=main)](https://codecov.io/github/gpuslave/simple-rag)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

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
just coverage              # run offline tests with an 85% coverage floor
just lint                  # check Python lint rules and formatting
just format                # apply safe lint fixes and formatting
just typecheck             # run strict static type checks
just doctor                # run live readiness checks
just doctor rag.toml       # use another configuration file
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

Provider routing under `[generation.provider]` is optional. Omitting the table leaves RouterAI routing unrestricted.

Architecture terminology is in [CONTEXT.md](CONTEXT.md); decisions are in [docs/adr](docs/adr).
