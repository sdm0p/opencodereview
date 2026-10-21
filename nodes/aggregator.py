from __future__ import annotations

import logging
import os
from pathlib import Path

from llm_factory import create_llm
from nodes.finding_quality import calibrate_findings, review_coverage
from pydantic import BaseModel, Field

from state import Finding, OpenCodeReviewState, Severity, Verdict

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.0

# Prompt version — tied to the file on disk
PROMPT_VERSION = "v1"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / f"aggregator_{PROMPT_VERSION}.txt"

# Findings below this confidence are dropped before the LLM step
MIN_CONFIDENCE = 0.25
# Files/line pairs within this many lines are considered duplicates
LINE_TOLERANCE = 5

# Default to rule-based aggregation (skips expensive LLM critic call).
# Set AGGREGATOR_USE_LLM_CRITIC=true in env to re-enable the LLM pass.
USE_LLM_CRITIC = os.environ.get("AGGREGATOR_USE_LLM_CRITIC", "").lower() in ("1", "true", "yes")


# ─── Load prompt from file ──────────────────────────────────────────────────


def _load_prompt() -> str:
    """Load the system prompt from the versioned prompt file."""
    try:
        return PROMPT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file %s not found — using fallback", PROMPT_FILE)
        return (
            "You are a senior engineer acting as a critic of a code review. "
            "Filter out noise, deduplicate findings, and produce a verdict. "
            "Return the filtered list of findings plus your verdict."
        )


SYSTEM_PROMPT = _load_prompt()


# ─── LLM output schema ──────────────────────────────────────────────────────


class AggregatedReview(BaseModel):
    """Final, noise-filtered findings and overall verdict."""
    findings: list[Finding] = Field(
        default_factory=list,
        description="Deduplicated, noise-filtered findings worth a senior engineer's time.",
    )
    verdict: Verdict = Field(
        description="Overall PR verdict after considering all findings.",
    )


# ─── Pre-LLM heuristic deduplication ────────────────────────────────────────


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    """Heuristic dedup: group by ``(file_path, line_start)`` with a line
    tolerance, keeping the highest-confidence finding per group."""
    if not findings:
        return []

    # Sort by (file, line_start) so we can iterate and merge neighbours
    sorted_fs = sorted(findings, key=lambda f: (f.file_path, f.line_start))

    merged: list[Finding] = []
    current: Finding = sorted_fs[0]

    for f in sorted_fs[1:]:
        same_file = f.file_path == current.file_path
        close_lines = abs(f.line_start - current.line_start) <= LINE_TOLERANCE
        same_claim = " ".join(f.comment.lower().split()) == " ".join(current.comment.lower().split())
        if same_file and close_lines and f.kind == current.kind and f.category == current.category and same_claim:
            # Keep the higher-confidence finding, merge comments
            if f.confidence > current.confidence:
                current = f
            if current.confidence < 0.9:
                current.confidence = round(
                    (current.confidence + f.confidence) / 2, 2
                )
            # Prefer the specific comment over a generic one
            if len(f.comment) > len(current.comment):
                current.comment = f.comment
            if f.suggested_fix and not current.suggested_fix:
                current.suggested_fix = f.suggested_fix
            # Escalate severity if either finding is more severe
            current.severity = _more_severe(current.severity, f.severity)
        else:
            merged.append(current)
            current = f
    merged.append(current)

    # Drop low-confidence noise
    merged = [f for f in merged if f.confidence >= MIN_CONFIDENCE]

    return merged


_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


def _more_severe(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_ORDER.get(a, 5) < _SEVERITY_ORDER.get(b, 5) else b


# ─── Rule-based fallback verdict ────────────────────────────────────────────


def _rule_verdict(findings: list[Finding]) -> Verdict:
    """Build a verdict from deduplicated findings without an LLM call."""
    eligible = [f for f in findings if f.kind != "test_suggestion" and f.evidence_status == "source_grounded"]
    suggestions = sum(f.kind == "test_suggestion" for f in findings)
    unverified = sum(f.kind != "test_suggestion" and f.evidence_status != "source_grounded" for f in findings)
    counts = {s: 0 for s in Severity}
    for f in eligible:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    critical = counts[Severity.CRITICAL]
    high = counts[Severity.HIGH]

    if critical > 0:
        recommendation = "block"
        approved = False
    elif high > 0:
        recommendation = "request_changes"
        approved = False
    elif eligible or suggestions or unverified:
        recommendation = "comment"
        approved = True
    else:
        recommendation = "approve"
        approved = True

    # Score: 10 - penalty based on severity
    score = max(0.0, min(10.0, 10.0 - critical * 4.0 - high * 2.0 - counts.get(Severity.MEDIUM, 0) * 0.5))

    severity_counts = {s.value: counts.get(s, 0) for s in Severity}

    return Verdict(
        summary=_rule_summary(recommendation, severity_counts) + (f" {suggestions} test suggestion(s), {unverified} unverified claim(s) are excluded from the score." if suggestions or unverified else ""),
        approved=approved,
        overall_score=round(score, 1),
        critical_count=critical,
        high_count=high,
        recommendation=recommendation,
        score_explanation="Risk estimate from source-grounded findings, not answer correctness or proof of safety. Test suggestions and claims without source grounding do not lower the score.",
        score_breakdown={"starting_score": 10, "critical_penalty": critical * 4, "high_penalty": high * 2,
                         "medium_penalty": counts[Severity.MEDIUM] * .5, "scored_findings": len(eligible),
                         "test_suggestions_excluded": suggestions, "unverified_claims_excluded": unverified},
    )


def _rule_summary(recommendation: str, counts: dict) -> str:
    parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
    if not parts:
        return "No source-grounded defects detected. This does not establish that the PR is safe."
    body = ", ".join(parts)
    if recommendation == "block":
        return f"Blocking: {body} — critical issues must be resolved."
    elif recommendation == "request_changes":
        return f"Changes requested: {body} — high-severity issues should be addressed."
    elif recommendation == "comment":
        return f"Minor concerns: {body} — consider addressing for best practices."
    return f"Approved: {body} — no significant issues."


# ─── Prompt builder ─────────────────────────────────────────────────────────


def _build_prompt(findings: list[Finding]) -> str:
    lines = [
        "Below are the raw findings from automated code reviewers "
        "(correctness, security, test-coverage).",
        "",
        "Filter out noise and return only the findings a senior engineer "
        "would actually leave on this PR.",
        "",
        "## Raw findings",
    ]
    for i, f in enumerate(findings, 1):
        lines.append(f"\n--- Finding {i} ---")
        lines.append(f"  File:      {f.file_path}")
        lines.append(f"  Lines:     {f.line_start}–{f.line_end}")
        lines.append(f"  Severity:  {f.severity.value}")
        lines.append(f"  Category:  {f.category}")
        lines.append(f"  Comment:   {f.comment}")
        lines.append(f"  Confidence: {f.confidence}")
        if f.suggested_fix:
            lines.append(f"  Fix:       {f.suggested_fix[:200]}")

    lines.append("\nNow apply the deduplication, noise-filtering, and verdict rules.")
    return "\n".join(lines)


# ─── Node ────────────────────────────────────────────────────────────────────


def aggregator_node(state: OpenCodeReviewState) -> dict:
    """Deduplicate, filter, and aggregate all findings into a final verdict.

    Returns state updates for ``findings`` (overwritten with the filtered
    list) and ``verdict`` (set-once by this node).

    Two-phase approach:
      1. Pre-LLM heuristic deduplication (groups by file/line proximity).
      2. Rule-based verdict by default (fast, ~10ms), or optional LLM critic
         when ``USE_LLM_CRITIC`` is enabled.
    """
    coverage = review_coverage(state)
    calibrated = calibrate_findings(state.findings, state)
    failed = [stage for stage in state.review_stages if stage.get("status") == "failed"]
    if failed:
        names = ", ".join(stage["reviewer"] for stage in failed)
        return {"review_coverage": coverage, "final_findings": _deduplicate(calibrated), "verdict": Verdict(
            summary=f"Review incomplete: {names} failed. Findings are partial; no approval or quality score is available.",
            approved=False, overall_score=None, recommendation="incomplete",
            score_explanation="No overall risk score is available because one or more reviewers failed. Remaining findings are partial.",
        )}
    raw_findings = calibrated
    if not raw_findings:
        logger.info("No findings to aggregate — producing empty verdict")
        return {"review_coverage": coverage, "final_findings": [], "verdict": _rule_verdict([])}

    # -- Phase 1: Heuristic dedup --------------------------------------------
    deduped = _deduplicate(raw_findings)
    logger.info(
        "Dedup heuristic: %d raw → %d merged",
        len(raw_findings), len(deduped),
    )

    # -- Phase 2: Rule-based (default, fast) or LLM critic (opt-in) ----------
    if USE_LLM_CRITIC:
        try:
            llm = create_llm(endpoint=state.endpoint or None)
        except ValueError as exc:
            logger.warning("LLM critic unavailable — using rule-based: %s", exc)
            llm = None

        if llm is not None:
            logger.info(
                "Invoking aggregator LLM critic (prompt_version=%s) …",
                PROMPT_VERSION,
            )
            try:
                structured_llm = llm.with_structured_output(AggregatedReview)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_prompt(deduped)},
                ]
                result: AggregatedReview = structured_llm.invoke(messages)  # type: ignore[assignment]
            except Exception as exc:
                logger.warning("LLM critic failed — falling back to rule-based: %s", exc)
                result = None

            if result is not None:
                final_findings = calibrate_findings(result.findings, state)
                verdict = _rule_verdict(final_findings)
                logger.info(
                    "LLM aggregator: %d deduped → %d final, recommendation=%s",
                    len(deduped), len(final_findings),
                    verdict.recommendation,
                )
                return {"review_coverage": coverage, "final_findings": final_findings, "verdict": verdict}

    # Rule-based path (default) — zero LLM overhead
    logger.info("Using rule-based aggregation (fast path)")
    final_findings = deduped
    verdict = _rule_verdict(deduped)

    return {
        "review_coverage": coverage,
        "final_findings": final_findings,  # plain list, overwrites
        "verdict": verdict,                # set-once, no reducer needed
    }
