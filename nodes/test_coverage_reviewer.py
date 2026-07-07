from __future__ import annotations

import logging
import os

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from state import Finding, OpenCodeReviewState

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 6_000
MAX_FILE_CONTENT_CHARS = 4_000


# ─── Wrapper schema ──────────────────────────────────────────────────────────


class TestCoverageReview(BaseModel):
    """Test-coverage gaps found during code review."""
    findings: list[Finding] = Field(
        default_factory=list,
        description="Test-coverage issues found in the PR diff.",
    )


# ─── Prompt ─────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a code reviewer focused on **test coverage**.  Analyse the
PR diff and determine whether the changed logic has adequate test coverage.

Look for:

1. **New logic without tests** — new functions, classes, or modules added in
   this diff that have no corresponding test file or test function.
2. **Modified behaviour without updated assertions** — existing logic that was
   changed in a semantically meaningful way (not just refactoring) but the
   corresponding tests were left unchanged, meaning they now pass vacuously or
   no longer cover the modified path.
3. **Edge-case branches left untested** — new `if`/`elif` branches, `try`/
   `except` handlers, or `match`/`case` arms that are not exercised by any
   test in the diff.
4. **Error / failure paths** — new error-return or exception-throwing paths
   that have no test asserting the error behaviour.
5. **Configuration / environment changes** — changes to config files,
   environment variables, or feature flags that affect behaviour but are not
   validated in tests.

For each gap return a `Finding` with:
- `file_path`: the source file with the gap (not the test file).
- `line_start` / `line_end`: best guess of the affected lines.
- `severity`: `high` (entirely new untested logic), `medium` (modified path
  untested), `low` (edge case not covered), `info` (suggestion).
- `category`: always `"test_coverage"`.
- `comment`: explain what changed and why it needs a test.
- `confidence`: 0.0–1.0.
- `suggested_fix`: what a good test should cover (not full code, just
  the scenario to test).

**Do NOT** comment on:
- Purely cosmetic or refactoring changes (rename, extract method).
- Third-party dependencies or generated files.
- Changes to test files themselves (they are the solution, not the problem).
- Correctness of the logic itself (handled by correctness reviewer).

Return an empty findings list if every changed path has adequate test coverage.
"""


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
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping test-coverage review")
        return {}

    if not state.diff:
        logger.info("No diff to review — skipping test-coverage review")
        return {}

    llm = ChatGroq(model=GROQ_MODEL, temperature=DEFAULT_TEMPERATURE, api_key=api_key)
    structured_llm = llm.with_structured_output(TestCoverageReview)

    user_prompt = _build_prompt(state)
    logger.info(
        "Invoking test-coverage reviewer (diff=%s, context=%d chunks)",
        _human_size(len(state.diff or "")),
        len(state.context_chunks),
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
