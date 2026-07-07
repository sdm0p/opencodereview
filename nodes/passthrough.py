from __future__ import annotations

import logging

from state import OpenCodeReviewState

logger = logging.getLogger(__name__)


def passthrough_node(state: OpenCodeReviewState) -> dict:
    """Log the current state and return unchanged.

    This is a placeholder node that proves the graph wiring and checkpointing
    work before any real review logic is added.
    """
    logger.info("=== Passthrough Node — %s#%d ===", state.repo, state.pr_number)
    logger.info("Files: %d  |  Chunks: %d  |  Findings: %d",
                len(state.changed_files),
                len(state.context_chunks),
                len(state.findings))
    logger.info("Verdict: %s", state.verdict.model_dump() if state.verdict else None)

    # Return empty dict — no state mutations in this placeholder
    return {}
