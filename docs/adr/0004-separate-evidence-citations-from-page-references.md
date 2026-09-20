# ADR 0004: Separate evidence citations from page references

Status: accepted

## Decision

Keep chunk-level evidence citations as the authoritative provenance for validation and audit, and derive page references for user-facing output. Page references group evidence from the same canonical PDF path and viewer page, preserve first document appearance, and order pages numerically within each document.

## Consequences

CLI and JSON consumers receive stable, non-duplicated page references without losing chunk scores, excerpts, or source IDs. Output adapters must render the validated page references rather than reconstructing them independently.
