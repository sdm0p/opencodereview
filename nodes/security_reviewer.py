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


class SecurityReview(BaseModel):
    """Security issues found during code review."""
    findings: list[Finding] = Field(
        default_factory=list,
        description="Security issues found in the PR diff.",
    )


# ─── Prompt ─────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a security-focused code reviewer.  Analyse the PR diff and
changed files for security vulnerabilities ONLY.  Look for:

1. **Injection** — SQL, NoSQL, command, template (SSTI), LDAP, XSS, path
   traversal via unsanitized user input.
2. **Secrets / credentials** — hardcoded API keys, tokens, passwords, or
   connection strings that should use environment variables or a secret store.
3. **Unsafe deserialization** — `pickle.loads`, `yaml.load(…)` without
   SafeLoader, `eval`, `exec`, or `marshal` on untrusted data.
4. **Authentication / authorisation** — missing or broken access control
   checks, privilege escalation paths, session fixation, weak password hashing.
5. **Input validation** — missing or insufficient validation of user-supplied
   data that could lead to buffer overflows, integer overflows, or format-string
   bugs.
6. **Cryptographic misuse** — weak algorithms (MD5, SHA1 for signatures),
   hardcoded IVs/nonces, ECB mode, insufficient key lengths, missing TLS.
7. **Information disclosure** — stack traces leaked to users, verbose error
   messages exposing internals, debug endpoints left enabled.

**Do NOT** comment on:
- Code style, formatting, naming conventions.
- Performance (unless it creates a denial-of-service vector).
- General logic errors or edge cases (handled by correctness reviewer).

For each issue return a `Finding` with:
- `file_path`, `line_start`, `line_end`: location in the diff.
- `severity`: `critical` (remotely exploitable), `high` (privilege escalation),
  `medium` (information disclosure), `low` (defence in depth), or `info`.
- `category`: always `"security"`.
- `comment`: concise, actionable explanation.
- `confidence`: 0.0–1.0.
- `suggested_fix`: concrete code snippet if applicable.

Return an empty findings list if no security issues are found.
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


def security_reviewer_node(state: OpenCodeReviewState) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping security review")
        return {}

    if not state.diff:
        logger.info("No diff to review — skipping security review")
        return {}

    llm = ChatGroq(model=GROQ_MODEL, temperature=DEFAULT_TEMPERATURE, api_key=api_key)
    structured_llm = llm.with_structured_output(SecurityReview)

    user_prompt = _build_prompt(state)
    logger.info(
        "Invoking security reviewer (diff=%s, context=%d chunks)",
        _human_size(len(state.diff or "")),
        len(state.context_chunks),
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        result: SecurityReview = structured_llm.invoke(messages)  # type: ignore[assignment]
    except Exception as exc:
        logger.warning("Security review LLM call failed: %s", exc)
        return {}

    if result.findings:
        logger.info("Security review found %d issue(s)", len(result.findings))
        for f in result.findings:
            logger.debug("  [%.2f] %s:%d–%d — %s", f.confidence, f.file_path, f.line_start, f.line_end, f.comment[:80])
    else:
        logger.info("Security review found no issues")

    return {"findings": result.findings}


def _human_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KiB"
    else:
        return f"{bytes_ / (1024 * 1024):.1f} MiB"
