# ADR 0001: Modular monolith using Hexagonal Architecture

Status: accepted

## Context

The application must remain testable without RouterAI, Qdrant, PyMuPDF, LangChain, or a particular user interface. Those technologies will evolve independently from the RAG use cases and domain vocabulary.

## Decision

Build one Python CLI application using Hexagonal Architecture, also called Ports and Adapters.

The inside of the application consists of domain models, use cases, and project-owned ports. The outside consists of technologies and delivery mechanisms:

- The CLI and automated tests are driving adapters that invoke application use cases.
- RouterAI, Qdrant, PyMuPDF, LangChain, and filesystem integrations are driven adapters behind project-owned ports.
- `cli.py` is the composition root that selects and wires concrete adapters.

Dependencies point inward. Domain and application modules must not import infrastructure modules. Public domain and port contracts must not expose third-party framework or SDK types. Ports describe purposeful application conversations rather than mirroring vendor APIs.

Adapters translate between external representations and project-owned types. Multiple adapters, including deterministic fakes, may satisfy the same port.

## Consequences

The application can be tested in isolation and its domain contract can remain stable while delivery, extraction, retrieval, storage, or model libraries change. New external technologies are added as adapters rather than imported into use cases.

The cost is explicit ports, dependency wiring, test fakes, and mapping at infrastructure boundaries. Architectural tests enforce the dependency direction.

## Reference

- [Hexagonal Architecture: the original Ports and Adapters article](https://alistair.cockburn.us/hexagonal-architecture/)
