# ADR 0003: Strict claim-level citation validation

Status: accepted

## Decision

Use only generation models whose RouterAI endpoints advertise strict structured outputs. Require each factual claim to reference retrieved request-local source IDs, validate the complete response, and never render an invalid answer.

## Consequences

Answers fail closed when model output or citations are malformed. Model choice is narrower, but printed page citations can be traced to retrieved evidence.

