from __future__ import annotations

import logging
import os
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from state import Finding, OpenCodeReviewState, Severity

logger = logging.getLogger(__name__)

# Model used on Groq — good at function calling / structured output
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0

# Max characters of context to include in the prompt (soft limit)
MAX_CONTEXT_CHARS = 6_000
# Max changed-file content to include per file
MAX_FILE_CONTENT_CHARS = 4_000


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


# ─── Prompt template ────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a seasoned code reviewer focused exclusively on **correctness**.

Analyse the PR diff and changed files below.  Look ONLY for:

1. **Logic errors** — incorrect boolean conditions, wrong operators, inverted
   if/else branches, incorrect variable shadowing, wrong function calls.
2. **Missed edge cases** — empty collections, None/null values, zero-length
   inputs, singleton lists, boundary values, negative numbers, overflow.
3. **Off-by-one errors** — loop bounds, slice indices, fencepost mistakes.
4. **Missing null/None checks** — unchecked return values, unchecked
   parameters that can be None, missing `is None` / `is not None` guards.
5. **Resource / state mismanagement** — unclosed handles, leaked connections,
   double-close, use-after-free, invalidated iterators.
6. **Concurrency hazards in the diff** — shared-mutation without locks, race
   windows, incorrect locking order.

**Do NOT** comment on:
- Code style, formatting, naming conventions, or readability.
- Performance (unless it's a correctness issue like an infinite loop).
- Security (that is handled by a separate reviewer).
- Tests being absent (unless the diff itself adds broken tests).

For each issue return a `Finding` with:
- `file_path`: the file the issue is in.
- `line_start` / `line_end`: the relevant line range (best guess from the diff).
- `severity`: `critical` (will definitely break), `high` (very likely to break),
  `medium` (edge-case bug), `low` (minor, unlikely to trigger), or `info`
  (informational).
- `category`: always `"correctness"`.
- `comment`: a concise, specific explanation of the bug.
- `confidence`: 0.0–1.0 reflecting how sure you are.
- `suggested_fix`: a concrete code snippet of how to fix it, if applicable.

If you find NO correctness issues, return an empty findings list.
"""


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
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping correctness review")
        return {}

    if not state.diff:
        logger.info("No diff to review — skipping correctness review")
        return {}

    # --- Build the LLM -------------------------------------------------------
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        api_key=api_key,
    )
    structured_llm = llm.with_structured_output(CorrectnessReview)

    # --- Build the prompt ----------------------------------------------------
    user_prompt = _build_prompt(state)
    logger.info(
        "Invoking correctness reviewer (diff=%s, context=%d chunks, prompt=%s)",
        _human_size(len(state.diff or "")),
        len(state.context_chunks),
        _human_size(len(user_prompt)),
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
