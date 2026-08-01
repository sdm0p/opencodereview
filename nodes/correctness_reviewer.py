from __future__ import annotations

import logging
import os
from pathlib import Path

from llm_factory import create_llm
from pydantic import BaseModel, Field

from state import Finding, OpenCodeReviewState, Severity

logger = logging.getLogger(__name__)

# Model used on Groq — good at function calling / structured output
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0

# Prompt version — tied to the file on disk
PROMPT_VERSION = "v1"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / f"correctness_{PROMPT_VERSION}.txt"

# Max characters of context to include in the prompt (soft limit)
MAX_CONTEXT_CHARS = 6_000
# Max changed-file content to include per file
MAX_FILE_CONTENT_CHARS = 4_000


# ─── Load prompt from file ──────────────────────────────────────────────────


def _load_prompt() -> str:
    """Load the system prompt from the versioned prompt file."""
    try:
        return PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file %s not found — using fallback", PROMPT_FILE)
        return (
            "You are a seasoned code reviewer focused exclusively on correctness. "
            "Analyse the PR diff and look for logic errors, edge cases, "
            "off-by-one errors, missing null checks, resource mismanagement, "
            "and concurrency hazards. Return a list of findings."
        )


SYSTEM_PROMPT = _load_prompt()


# ─── Wrapper schema for structured output ───────────────────────────────────


class CorrectnessReview(BaseModel):
    """A list of correctness issues found during code review.

    The LLM MUST populate this with every correctness-relevant finding in the
    diff.  Return an empty list if no correctness issues are found.
    """
    findings: list[Finding] = Field(
        default_factory=list,
        description=(
            "Correctness issues found in the PR diff. "
            "Each issue must reference a specific file and line range."
        ),
    )


# ─── Prompt builder ─────────────────────────────────────────────────────────


def _build_prompt(state: OpenCodeReviewState) -> str:
    """Build the user-facing prompt from the current graph state."""
    lines: list[str] = []

    lines.append("## Diff\n```diff")
    lines.append(state.diff or "")
    lines.append("```\n")

    # Changed files with full content (truncated per file)
    if state.changed_files:
        lines.append("## Changed files\n")
        for cf in state.changed_files:
            header = f"### {cf.path} ({cf.status})"
            lines.append(header)
            content = cf.content or ""
            if len(content) > MAX_FILE_CONTENT_CHARS:
                content = content[:MAX_FILE_CONTENT_CHARS] + "\n… (truncated)"
            lines.append(f"```\n{content}\n```\n")

    # Context chunks (truncated total)
    if state.context_chunks:
        lines.append("## Retrieved context\n")
        ctx_chars = 0
        for chunk in state.context_chunks:
            snippet = chunk.content
            if ctx_chars + len(snippet) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - ctx_chars
                if remaining > 200:
                    lines.append(f"> {snippet[:remaining]}… (truncated)")
                break
            if chunk.source == "past_pr":
                lines.append(f"> [Past PR — {chunk.file_path}]: {snippet}")
            else:
                lines.append(f"> [{chunk.file_path}]: {snippet}")
            ctx_chars += len(snippet)

    return "\n".join(lines)


# ─── Node ────────────────────────────────────────────────────────────────────


def correctness_reviewer_node(state: OpenCodeReviewState) -> dict:
    """Analyse the diff and context chunks for correctness issues using an LLM.

    Returns state updates for ``findings`` (accumulated via the ``add``
    reducer).
    """
    # --- Sanity checks -------------------------------------------------------
    if not state.diff:
        logger.info("No diff to review — skipping correctness review")
        return {}

    # --- Build the LLM (Gemini primary → Groq fallback) -----------------------
    try:
        llm = create_llm(endpoint=state.endpoint or None)
    except ValueError as exc:
        logger.warning("LLM unavailable — skipping correctness review: %s", exc)
        return {}
    structured_llm = llm.with_structured_output(CorrectnessReview)

    # --- Build the prompt ----------------------------------------------------
    user_prompt = _build_prompt(state)
    logger.info(
        "Invoking correctness reviewer (diff=%s, context=%d chunks, prompt=%s, prompt_version=%s)",
        _human_size(len(state.diff or "")),
        len(state.context_chunks),
        _human_size(len(user_prompt)),
        PROMPT_VERSION,
    )

    # --- Invoke --------------------------------------------------------------
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        result: CorrectnessReview = structured_llm.invoke(messages)  # type: ignore[assignment]
    except Exception as exc:
        logger.warning("Correctness review LLM call failed: %s", exc)
        return {}

    # --- Log and return findings ---------------------------------------------
    if result and result.findings:
        logger.info(
            "Correctness review found %d issue(s)",
            len(result.findings),
        )
        for f in result.findings:
            logger.debug(
                "  [%.2f] %s:%d–%d — %s",
                f.confidence, f.file_path, f.line_start, f.line_end,
                f.comment[:80],
            )
    else:
        logger.info("Correctness review found no issues")

    return {
        "findings": result.findings,
    }


# ─── Utility ─────────────────────────────────────────────────────────────────


def _human_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KiB"
    else:
        return f"{bytes_ / (1024 * 1024):.1f} MiB"
