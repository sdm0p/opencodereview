from __future__ import annotations

import logging
import os
from pathlib import Path

from llm_factory import create_llm
from pydantic import BaseModel, Field

from state import Finding, OpenCodeReviewState

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 6_000
MAX_FILE_CONTENT_CHARS = 4_000

# Prompt version — tied to the file on disk
PROMPT_VERSION = "v1"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / f"test_coverage_{PROMPT_VERSION}.txt"


# ─── Load prompt from file ──────────────────────────────────────────────────


def _load_prompt() -> str:
    """Load the system prompt from the versioned prompt file."""
    try:
        return PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file %s not found — using fallback", PROMPT_FILE)
        return (
            "You are a code reviewer focused on test coverage. "
            "Analyse the PR diff and determine whether the changed logic "
            "has adequate test coverage. Return a list of findings."
        )


SYSTEM_PROMPT = _load_prompt()


# ─── Wrapper schema ──────────────────────────────────────────────────────────


class TestCoverageReview(BaseModel):
    """Test-coverage gaps found during code review."""
    findings: list[Finding] = Field(
        default_factory=list,
        description="Test-coverage issues found in the PR diff.",
    )


# ─── Prompt builder ─────────────────────────────────────────────────────────


def _build_prompt(state: OpenCodeReviewState) -> str:
    lines: list[str] = []

    lines.append("## Diff\n```diff")
    lines.append(state.diff or "")
    lines.append("```\n")

    if state.changed_files:
        lines.append("## Changed files\n")
        for cf in state.changed_files:
            lines.append(f"### {cf.path} ({cf.status})")
            content = cf.content or ""
            if len(content) > MAX_FILE_CONTENT_CHARS:
                content = content[:MAX_FILE_CONTENT_CHARS] + "\n… (truncated)"
            lines.append(f"```\n{content}\n```\n")

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
            prefix = "Past PR" if chunk.source == "past_pr" else chunk.file_path
            lines.append(f"> [{prefix}]: {snippet}")
            ctx_chars += len(snippet)

    return "\n".join(lines)


# ─── Node ────────────────────────────────────────────────────────────────────


def test_coverage_reviewer_node(state: OpenCodeReviewState) -> dict:
    if not state.diff:
        logger.info("No diff to review — skipping test-coverage review")
        return {}

    try:
        llm = create_llm()
    except ValueError:
        logger.warning("No LLM key set — skipping test-coverage review")
        return {}
    structured_llm = llm.with_structured_output(TestCoverageReview)

    user_prompt = _build_prompt(state)
    logger.info(
        "Invoking test-coverage reviewer (diff=%s, context=%d chunks, prompt_version=%s)",
        _human_size(len(state.diff or "")),
        len(state.context_chunks),
        PROMPT_VERSION,
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        result: TestCoverageReview = structured_llm.invoke(messages)  # type: ignore[assignment]
    except Exception as exc:
        logger.warning("Test-coverage review LLM call failed: %s", exc)
        return {}

    if result.findings:
        logger.info("Test-coverage review found %d issue(s)", len(result.findings))
        for f in result.findings:
            logger.debug("  [%.2f] %s:%d–%d — %s", f.confidence, f.file_path, f.line_start, f.line_end, f.comment[:80])
    else:
        logger.info("Test-coverage review found no issues")

    return {"findings": result.findings}


def _human_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KiB"
    else:
        return f"{bytes_ / (1024 * 1024):.1f} MiB"
