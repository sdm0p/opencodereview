#!/usr/bin/env python3
"""Evaluate the OpenCodeReview agent against ground-truth human review comments.

Usage
-----
    # Run evaluation (requires ground-truth dataset from fetcher.py)
    python eval_data/evaluate.py

    # Only process the first 2 PRs (quick smoke test)
    python eval_data/evaluate.py --max-prs 2

    # Show per-PR details (matched/unmatched findings)
    python eval_data/evaluate.py --verbose

    # Enable keyword-based semantic matching (bonus for keyword overlap)
    python eval_data/evaluate.py --match-keywords

    # Bypass the GITHUB_TOKEN guard (use with caution)
    python eval_data/evaluate.py --force

Input
-----
    ``eval_data/prs.jsonl`` -- ground-truth dataset produced by ``fetcher.py``.

Output
------
    ``eval_data/eval_results.json`` -- per-PR and aggregate metrics.

Notes
-----
    Human PR review comments don't carry explicit category labels, so
    category matching is approximated via keyword overlap when
    ``--match-keywords`` is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Optional

from langgraph.types import Command

# Add project root to sys.path so we can import graph/state
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import build_graph
from observability import langfuse_trace, _resolve_langfuse_trace_id
from state import ChangedFile, Finding

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

LINE_TOLERANCE = 3
DATA_PATH = os.path.join(os.path.dirname(__file__), "prs.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

# Common English stopwords (lightweight -- no dependency needed)
STOPWORDS: set[str] = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "not", "no", "nor",
    "or", "and", "but", "if", "in", "on", "at", "to", "for", "of",
    "by", "with", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too",
    "very", "just", "because", "also", "any", "get", "use", "using",
    "used", "like", "make", "made", "take", "know", "see", "look",
    "come", "want", "give", "tell", "work", "call", "try", "ask",
    "feel", "become", "leave", "put", "mean", "keep", "let", "begin",
    "seem", "help", "turn", "show", "play", "run", "move", "live",
    "hold", "bring", "happen", "write", "provide", "sit", "stand",
    "lose", "pay", "meet", "include", "continue", "set", "learn",
    "change", "lead", "follow", "stop", "create", "read", "allow",
    "add", "spend", "grow", "open", "win", "offer", "remember",
    "consider", "appear", "buy", "wait", "serve", "send", "expect",
    "build", "stay", "fall", "cut", "reach", "remain", "suggest",
    "raise", "pass", "sell", "require", "report", "decide", "pull",
    "push", "please", "thanks", "thank", "hi", "hello", "hey",
    "one", "two", "way", "thing", "things", "much", "many",
    "yes", "no", "sure", "right", "good", "great", "nice", "fine",
    "well", "bad", "wrong", "new", "old", "first", "last", "next",
    "previous", "still", "already", "even", "ever", "never", "always",
    "really", "actually", "basically", "essentially", "probably",
    "maybe", "perhaps", "quite", "pretty", "rather", "little",
    "big", "large", "small", "long", "short", "high", "low",
    "whole", "entire", "full", "empty", "part", "piece", "bit",
    "type", "kind", "sort", "way", "case", "example", "instance",
    "instead", "rather", "else", "otherwise", "well",
}


# --- Fuzzy Matching ----------------------------------------------------------


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a comment."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _lines_overlap(
    f_start: int, f_end: int, gt_line: int, tolerance: int = LINE_TOLERANCE,
) -> bool:
    """Check if a ground-truth line falls within a finding's line range
    (with tolerance)."""
    lo = min(f_start, f_end) - tolerance
    hi = max(f_start, f_end) + tolerance
    return lo <= gt_line <= hi


def _fuzzy_match(
    finding: Finding,
    ground_truth: list[dict],
    already_matched: set[int],
    use_keywords: bool = False,
) -> Optional[int]:
    """Find the best ground-truth comment that matches this agent finding.

    Matching criteria (in order):
    1. File path (exact match)
    2. Line proximity (within +/-LINE_TOLERANCE)
    3. Keyword overlap bonus (optional, via --match-keywords)

    Returns the index of the best match, or None.
    """
    best_idx: Optional[int] = None
    best_score = -1.0

    for i, gt in enumerate(ground_truth):
        if i in already_matched:
            continue

        # File must match (exact path)
        if finding.file_path != gt.get("file_path", ""):
            continue

        # Line proximity
        gt_line = gt.get("line", 0)
        if not _lines_overlap(finding.line_start, finding.line_end, gt_line):
            continue

        # Base score: line proximity (closer = better)
        line_distance = min(
            abs(finding.line_start - gt_line),
            abs(finding.line_end - gt_line),
        )
        score = 1.0 - (line_distance / (LINE_TOLERANCE + 1))
        assert 0.0 <= score <= 1.0

        # Keyword overlap bonus (optional)
        if use_keywords:
            f_kw = _extract_keywords(finding.comment)
            g_kw = _extract_keywords(gt.get("comment", ""))
            if g_kw:
                overlap = len(f_kw & g_kw) / len(g_kw)
                score += overlap * 0.5  # up to 0.5 bonus

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# --- Graph Runner ------------------------------------------------------------


def _run_graph_for_pr(
    entry: dict, graph,
) -> tuple[list[Finding], Optional[dict], list, Optional[str]]:
    """Run the full OpenCodeReview graph on a single PR.

    Returns ``(final_findings, verdict_dict_or_None, context_chunks, trace_id_or_None)``.
    """
    thread_id = str(uuid.uuid4())
    repo = entry.get("repo", "?")
    pr_number = entry.get("pr_number", 0)
    config = {"configurable": {"thread_id": f"eval-{thread_id}"}}

    # Build initial state from pre-fetched data to avoid API calls
    changed_files = [
        ChangedFile(**cf) for cf in entry.get("changed_files", [])
    ]
    initial_state = {
        "repo": entry["repo"],
        "pr_number": entry["pr_number"],
        "diff": entry.get("diff", ""),
        "changed_files": changed_files,
    }

    trace_id: Optional[str] = None
    context_chunks: list = []

    with langfuse_trace(
        trace_name=f"opencodereview/eval/{repo}#{pr_number}",
        tags=["eval", repo],
        session_id=thread_id,
    ) as handler:
        callbacks = []
        if handler:
            callbacks.append(handler)
        config["callbacks"] = callbacks

        # Run -- the graph will pause at human_approval (interrupt)
        try:
            list(graph.stream(initial_state, config))
        except Exception as exc:
            logger.warning("  Graph stream error: %s", exc)

        state = graph.get_state(config)
        tasks = state.tasks
        values = state.values

        context_chunks = values.get("context_chunks", [])

        if not tasks:
            # Graph completed without interrupt -- no findings to review
            trace_id = _resolve_langfuse_trace_id(handler)
            return (
                values.get("final_findings", []),
                values.get("verdict"),
                context_chunks,
                trace_id,
            )

        # Auto-approve to continue through the executor
        try:
            graph.invoke(Command(resume={"action": "approve"}), config)
        except Exception as exc:
            logger.warning("  Resume error: %s", exc)
            trace_id = _resolve_langfuse_trace_id(handler)
            return (
                values.get("final_findings", []),
                values.get("verdict"),
                context_chunks,
                trace_id,
            )

        final_state = graph.get_state(config)
        trace_id = _resolve_langfuse_trace_id(handler)
        return (
            final_state.values.get("final_findings", []),
            final_state.values.get("verdict"),
            final_state.values.get("context_chunks", []),
            trace_id,
        )


# --- Metrics -----------------------------------------------------------------


def _compute_metrics(
    findings: list[Finding],
    ground_truth: list[dict],
    use_keywords: bool = False,
) -> dict:
    """Compute TP / FP / FN and derived metrics."""
    matched: set[int] = set()
    tp = 0
    fp = 0

    for f in findings:
        match_idx = _fuzzy_match(f, ground_truth, matched, use_keywords)
        if match_idx is not None:
            tp += 1
            matched.add(match_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "agent_findings": len(findings),
        "human_comments": len(ground_truth),
    }


# --- Output ------------------------------------------------------------------


def _print_results(results: list[dict]) -> None:
    """Print a formatted evaluation results table."""
    print()
    print("=" * 100)
    hdr = f"{'PR':<40} {'Agent':>6} {'Human':>6} {'TP':>4} {'FP':>4} {'FN':>4}"
    hdr += f" {'Prec':>6} {'Recall':>6} {'F1':>6}"
    print(hdr)
    print("-" * 100)

    for r in results:
        label = f"{r['repo']}#{r['pr_number']}"
        print(
            f"{label:<40} {r['agent_findings']:>6} {r['human_comments']:>6} "
            f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
            f"{r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%}"
        )

    print("-" * 100)

    # Micro-average (pool all TP/FP/FN)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_agent = sum(r["agent_findings"] for r in results)
    total_human = sum(r["human_comments"] for r in results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    # Macro-average (average per-PR metrics, excluding PRs with no findings AND no comments)
    nonzero = [r for r in results if r["agent_findings"] > 0 or r["human_comments"] > 0]
    macro_p = sum(r["precision"] for r in nonzero) / len(nonzero) if nonzero else 0.0
    macro_r = sum(r["recall"] for r in nonzero) / len(nonzero) if nonzero else 0.0
    macro_f = sum(r["f1"] for r in nonzero) / len(nonzero) if nonzero else 0.0

    print(
        f"{'MICRO AVG':<40} {total_agent:>6} {total_human:>6} "
        f"{total_tp:>4} {total_fp:>4} {total_fn:>4} "
        f"{micro_p:>6.1%} {micro_r:>6.1%} {micro_f:>6.1%}"
    )
    print(
        f"{'MACRO AVG':<40} {' ':>6} {' ':>6} "
        f"{' ':>4} {' ':>4} {' ':>4} "
        f"{macro_p:>6.1%} {macro_r:>6.1%} {macro_f:>6.1%}"
    )
    print("=" * 100)
    print()


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenCodeReview against ground-truth human reviews",
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to JSONL dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-PR details (matched / unmatched findings)",
    )
    parser.add_argument(
        "--match-keywords", action="store_true",
        help="Enable keyword-based semantic matching bonus "
             "(human comments don't carry explicit category labels, so "
             "keyword overlap approximates category matching)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="IDs of PRs to skip (space-separated)",
    )
    parser.add_argument(
        "--max-prs", type=int, default=None,
        help="Max PRs to process (for quick smoke tests)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass GITHUB_TOKEN guard and proceed with posting "
             "(use with caution -- comments WILL be posted to real PRs)",
    )
    parser.add_argument(
        "--log-to-observability", action="store_true",
        help="Log evaluation scores to LangSmith/Langfuse for quality-over-time tracking",
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Compute RAGAS retrieval & generation metrics (context_precision, context_recall, "
             "faithfulness, answer_relevancy, mmr) in addition to standard precision/recall/F1",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] - %(message)s",
    )

    # -- Guard: GITHUB_TOKEN -> accidental posting --------------------------
    github_token_warning = (
        "WARNING: GITHUB_TOKEN is set! The executor will POST comments "
        "to real GitHub PRs.\n"
        "  Unset it for safe evaluation:\n"
        "    unset GITHUB_TOKEN\n"
        "  Pass --force to acknowledge and proceed anyway."
    )
    if "GITHUB_TOKEN" in os.environ and not args.force:
        print(f"\n{'=' * 60}\n  {github_token_warning}\n{'=' * 60}\n", file=sys.stderr)
        sys.exit(1)

    # --- Load dataset ---
    if not os.path.exists(args.data):
        print(
            f"Dataset not found at {args.data}. Run fetcher.py first:\n"
            f"  python eval_data/fetcher.py\n"
            f"Note: fetcher.py requires GITHUB_TOKEN for API access.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d PRs from %s", len(entries), args.data)

    if args.max_prs:
        entries = entries[: args.max_prs]

    # --- Build graph ---
    print("Building graph ... ", end="", flush=True)
    graph = build_graph(":memory:")
    print("OK")

    # --- Evaluate each PR ---
    results: list[dict] = []

    for idx, entry in enumerate(entries, 1):
        pr_id = entry["id"]
        if pr_id in args.skip:
            logger.info("[%d/%d] Skipping %s", idx, len(entries), pr_id)
            continue

        print(
            f"[{idx}/{len(entries)}] {entry['repo']}#{entry['pr_number']} "
            f"-- {entry.get('title', '')[:50]} ... ",
            end="", flush=True,
        )

        try:
            findings, verdict, context_chunks, trace_id = _run_graph_for_pr(entry, graph)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        metrics = _compute_metrics(
            findings, entry["ground_truth"],
            use_keywords=args.match_keywords,
        )
        metrics.update({
            "id": pr_id,
            "repo": entry["repo"],
            "pr_number": entry["pr_number"],
            "title": entry.get("title", ""),
            "trace_id": trace_id,
        })

        # ── Compute RAGAS retrieval & generation metrics (optional) ──
        if args.ragas and context_chunks:
            try:
                from eval_data.ragas_eval import compute_ragas_retrieval_scores

                query = entry.get("diff", "")[:8_000] or entry.get("title", "")
                contexts = [c.content for c in context_chunks if c.content]
                gt_text = " ".join(
                    g.get("comment", "")
                    for g in entry.get("ground_truth", [])[:5]
                ) if entry.get("ground_truth") else None

                # Build the generated answer text from findings + verdict
                answer_lines = []
                for f in findings:
                    answer_lines.append(
                        f"[{f.severity.value}] {f.file_path}:{f.line_start}-{f.line_end}: {f.comment}"
                    )
                if verdict:
                    answer_lines.append(
                        f"Verdict: {verdict.recommendation} (score={verdict.overall_score}/10) — {verdict.summary}"
                    )
                answer_text = "\n".join(answer_lines) if answer_lines else None

                ragas_scores = compute_ragas_retrieval_scores(
                    query=query,
                    retrieved_contexts=contexts,
                    ground_truth=gt_text or None,
                    answer=answer_text,
                )
                metrics.update(ragas_scores)
                print(
                    f"  RAGAS: "
                    + " | ".join(
                        f"{k}={v:.3f}"
                        for k, v in ragas_scores.items()
                        if v is not None
                    )
                )
            except Exception as exc:
                logger.warning("RAGAS computation failed: %s", exc)

        results.append(metrics)

        print(
            f"{metrics['agent_findings']} findings vs "
            f"{metrics['human_comments']} human "
            f"(P={metrics['precision']:.0%} R={metrics['recall']:.0%})",
        )

        if args.verbose:
            _print_verbose(entry, findings, metrics, use_keywords=args.match_keywords)

    # --- Print results ---
    _print_results(results)

    # --- Log to observability (optional) ---
    if args.log_to_observability and results:
        _log_to_observability(results)

    # --- Save ---
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {RESULTS_PATH}")


def _print_verbose(entry: dict, findings: list[Finding], metrics: dict, use_keywords: bool = False) -> None:
    """Print per-finding details for a single PR."""
    matched_gt: set[int] = set()

    print(f"\n  {'=' * 60}")
    print(f"  Agent findings ({metrics['agent_findings']}):")
    for f in findings:
        match_idx = _fuzzy_match(f, entry["ground_truth"], matched_gt, use_keywords)
        flag = "[MATCHED]" if match_idx is not None else "[MISSED]"
        if match_idx is not None:
            matched_gt.add(match_idx)
        print(
            f"    {flag} [{f.severity.value}] {f.file_path}:{f.line_start}"
            f"-{f.line_end} - {f.comment[:100]}",
        )

    print(f"  Human ground-truth ({metrics['human_comments']}):")
    for i, gt in enumerate(entry["ground_truth"]):
        flag = " [HIT]" if i in matched_gt else " [MISS]"
        print(
            f"    {flag} {gt['file_path']}:{gt['line']} - "
            f"{gt['comment'][:100]}",
        )
    print()


# ─── Quality-over-time: log to observability ───────────────────────────────


def _log_to_observability(results: list[dict]) -> None:
    """Log evaluation results to the active observability backend(s).

    Per-PR scores (f1, context_precision, etc.) are linked to each PR's
    pipeline trace via ``trace_id``, so they appear directly on the trace
    in the Langfuse UI.  Aggregate scores (micro_f1, pr_count, etc.) are
    logged without a trace since they summarise multiple runs.

    This function is called only when ``--log-to-observability`` is passed.
    """
    from observability import (
        is_langfuse_enabled,
        is_langsmith_enabled,
        log_langfuse_score,
    )

    # Compute aggregate metrics
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    # RAGAS metric keys available
    RAGAS_METRICS = [
        "context_precision", "context_recall",
        "faithfulness", "answer_relevancy", "mmr",
    ]

    # Check which RAGAS scores were computed
    has_ragas = bool(results and any(k in results[0] for k in RAGAS_METRICS))
    ragas_avgs: dict[str, float | None] = {}
    if has_ragas:
        for metric in RAGAS_METRICS:
            values = [r[metric] for r in results if r.get(metric) is not None]
            ragas_avgs[metric] = sum(values) / len(values) if values else None

    log_parts = [f"micro F1={micro_f:.3f}"]
    if has_ragas:
        for metric, avg in ragas_avgs.items():
            if avg is not None:
                log_parts.append(f"RAGAS_{metric}={avg:.3f}")
    log_parts.append(f"PRs={len(results)}")
    log_parts.append(f"LangSmith={is_langsmith_enabled()}")
    log_parts.append(f"Langfuse={is_langfuse_enabled()}")

    logger.info(
        "Logging eval results to observability (%s)",
        ", ".join(log_parts),
    )

    # ── Langfuse: create scores via SDK ──────────────────────────────────
    if is_langfuse_enabled():
        try:
            # Use the v4 shared singleton so per-PR scores logged via
            # log_langfuse_score (also get_client()) share one client and
            # are flushed together below.
            from langfuse import get_client

            lf = get_client()
            run_name = f"eval-{datetime.now():%Y%m%d-%H%M%S}"

            # Log per-PR scores (linked to each PR's pipeline trace)
            for r in results:
                pr_trace_id = r.get("trace_id")
                pr_comment = f"{r.get('repo', '?')}#{r.get('pr_number', '?')}"
                pr_metadata = {
                    "pr_id": r.get("id"),
                    "repo": r.get("repo"),
                    "pr_number": r.get("pr_number"),
                }

                # Standard eval metrics
                lf.create_score(
                    name="f1",
                    value=r["f1"],
                    trace_id=pr_trace_id,
                    comment=pr_comment + f" — P={r['precision']:.3f} R={r['recall']:.3f}",
                    metadata=pr_metadata,
                )
                # Per-PR RAGAS scores — logged via the shared scoring path with
                # the canonical ``ragas_`` prefix so trace-linking + error
                # visibility apply uniformly (same as app.py/main.py).
                for ragas_key in ("context_precision", "context_recall",
                                  "faithfulness", "answer_relevancy", "mmr"):
                    if ragas_key in r and r[ragas_key] is not None:
                        log_langfuse_score(
                            name=f"ragas_{ragas_key}",
                            value=r[ragas_key],
                            comment=pr_comment,
                            trace_id=pr_trace_id,
                        )

            # Log aggregate scores (no trace_id — spans multiple runs)
            lf.create_score(name="micro_f1", value=micro_f)
            lf.create_score(name="micro_precision", value=micro_p)
            lf.create_score(name="micro_recall", value=micro_r)
            lf.create_score(name="pr_count", value=len(results))

            # Log aggregate RAGAS scores
            if has_ragas:
                for metric, avg in ragas_avgs.items():
                    if avg is not None:
                        lf.create_score(name=metric, value=avg)

            # Flush and shutdown to ensure scores are sent before process exits
            lf.flush()
            lf.shutdown()
            logger.info("Langfuse scores logged (run=%s)", run_name)
        except Exception as exc:
            logger.warning("Failed to log to Langfuse: %s", exc)

    # ── LangSmith: scores are automatic via graph tracing ────────────────
    # Every graph.invoke()/graph.stream() call is already traced by LangSmith.
    # We just log the aggregate for visibility.
    if is_langsmith_enabled():
        logger.info(
            "LangSmith tracing active — eval scores tracked in traced runs. "
            "Check your LangSmith project for detailed per-PR traces."
        )


if __name__ == "__main__":
    main()
