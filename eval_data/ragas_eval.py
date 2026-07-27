#!/usr/bin/env python3
"""RAGAS-based evaluation of retrieval quality for OpenCodeReview.

Computes **context_precision** and **context_recall** metrics from the
RAGAS framework (https://docs.ragas.io/) on the retrieved context chunks
produced by the ``retrieval_node``.

These metrics answer:
- **Context Precision**: Are the most relevant code chunks ranked first?
- **Context Recall**: Does the retrieved set contain all the information
  needed for the reviewers to produce accurate findings?

Usage
-----
    from eval_data.ragas_eval import compute_ragas_retrieval_scores

    score = compute_ragas_retrieval_scores(
        query="diff of PR ...",
        retrieved_contexts=["chunk1 text", "chunk2 text", ...],
        ground_truth="Human review comment text ...",
    )
    # score == {"context_precision": 0.85, "context_recall": 0.72}

Lazy imports minimise overhead when RAGAS is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Lazy RAGAS imports with graceful fallback ──────────────────────────────

_RAGAS_AVAILABLE: bool | None = None


def _check_ragas() -> bool:
    """Return True if RAGAS and litellm are available."""
    global _RAGAS_AVAILABLE
    if _RAGAS_AVAILABLE is not None:
        return _RAGAS_AVAILABLE
    try:
        import ragas  # noqa: F401
        import litellm  # noqa: F401
        _RAGAS_AVAILABLE = True
    except ImportError:
        logger.warning(
            "ragas or litellm not installed. Install with: "
            "pip install ragas litellm"
        )
        _RAGAS_AVAILABLE = False
    return _RAGAS_AVAILABLE


def _get_evaluator_llm():
    """Return a RAGAS-compatible LLM, reusing the project's LLM factory.

    Follows the same priority as the main pipeline:
    1. Google Gemini (``gemini-3.1-flash-lite``) via ``GEMINI_API_KEY``
    2. Groq (``llama-3.3-70b-versatile``) via ``GROQ_API_KEY``

    RAGAS v0.3+ uses ``LangchainLLMWrapper`` to wrap LangChain models.
    """
    from ragas.llms import LangchainLLMWrapper

    try:
        from llm_factory import create_llm
        llm = create_llm(temperature=0)
        logger.info(
            "RAGAS evaluator: using %s",
            type(llm).__name__,
        )
        return LangchainLLMWrapper(llm)
    except ValueError as exc:
        logger.warning("RAGAS evaluator: %s", exc)
        raise RuntimeError(
            "No LLM available for RAGAS scoring. "
            "Set GEMINI_API_KEY or GROQ_API_KEY."
        ) from exc


# ─── Score computation ─────────────────────────────────────────────────────


def compute_ragas_retrieval_scores(
    query: str,
    retrieved_contexts: list[str],
    ground_truth: Optional[str] = None,
    answer: Optional[str] = None,
) -> dict[str, float]:
    """Compute RAGAS retrieval & generation quality metrics.

    Computes up to five metrics depending on available inputs:

    +---------------------------+------------------------------------------+
    | Metric                   | Required inputs                          |
    +===========================+==========================================+
    | ``context_precision``     | query, retrieved_contexts                |
    | ``context_recall``        | query, retrieved_contexts, ground_truth  |
    | ``faithfulness``          | query, retrieved_contexts, answer        |
    | ``answer_relevancy``      | query, answer                            |
    | ``mmr`` (Mean Reciprocal  | query, retrieved_contexts                |
    | Rank)                     | (custom LLM-judged)                      |
    +---------------------------+------------------------------------------+

    Parameters
    ----------
    query : str
        The search query — in OpenCodeReview this is the PR diff text.
    retrieved_contexts : list[str]
        The code/document chunks returned by the retrieval node.
    ground_truth : str or None
        Ground-truth reference text (e.g. human review comment).  Required
        for ``context_recall``.
    answer : str or None
        The generated answer (e.g. concatenated review findings / verdict).
        Required for ``faithfulness`` and ``answer_relevancy``.

    Returns
    -------
    dict
        Mapping of metric name → score (0–1 range, higher is better).
        Possible keys: ``context_precision``, ``context_recall``,
        ``faithfulness``, ``answer_relevancy``, ``mmr``.

    Raises
    ------
    RuntimeError
        If RAGAS or a suitable LLM backend is not available.
    """
    if not _check_ragas():
        raise RuntimeError(
            "RAGAS is not installed. Run: pip install ragas litellm"
        )

    if not retrieved_contexts:
        logger.warning("No retrieved contexts — RAGAS scores set to 0.0")
        result: dict[str, float] = {"context_precision": 0.0, "mmr": 0.0}
        if ground_truth:
            result["context_recall"] = 0.0
        if answer:
            result["faithfulness"] = 0.0
            result["answer_relevancy"] = 0.0
        return result

    from ragas.dataset_schema import SingleTurnSample

    llm = _get_evaluator_llm()
    scores: dict[str, float] = {}

    # ── Context Precision ────────────────────────────────────────────
    try:
        from ragas.metrics.collections import ContextPrecision

        scorer = ContextPrecision(llm=llm)
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=retrieved_contexts,
        )
        result = scorer.score(sample)
        scores["context_precision"] = round(float(result.value), 4)
        logger.info("RAGAS context_precision: %.4f", scores["context_precision"])
    except Exception as exc:
        logger.warning("RAGAS context_precision failed: %s", exc)
        scores["context_precision"] = 0.0

    # ── Context Recall ───────────────────────────────────────────────
    if ground_truth:
        try:
            from ragas.metrics.collections import ContextRecall

            recall_scorer = ContextRecall(llm=llm)
            recall_sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
            recall_result = recall_scorer.score(recall_sample)
            scores["context_recall"] = round(float(recall_result.value), 4)
            logger.info("RAGAS context_recall: %.4f", scores["context_recall"])
        except Exception as exc:
            logger.warning("RAGAS context_recall failed: %s", exc)
            scores["context_recall"] = 0.0

    # ── Faithfulness ─────────────────────────────────────────────────
    if answer:
        try:
            from ragas.metrics import Faithfulness

            faithfulness_scorer = Faithfulness(llm=llm)
            faithful_sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=retrieved_contexts,
            )
            faithful_result = faithfulness_scorer.score(faithful_sample)
            scores["faithfulness"] = round(float(faithful_result.value), 4)
            logger.info("RAGAS faithfulness: %.4f", scores["faithfulness"])
        except Exception as exc:
            logger.warning("RAGAS faithfulness failed: %s", exc)
            scores["faithfulness"] = 0.0

    # ── Answer Relevancy (ResponseRelevancy) ─────────────────────────
    if answer:
        try:
            from ragas.metrics import ResponseRelevancy

            relevancy_scorer = ResponseRelevancy(llm=llm)
            relevancy_sample = SingleTurnSample(
                user_input=query,
                response=answer,
            )
            relevancy_result = relevancy_scorer.score(relevancy_sample)
            scores["answer_relevancy"] = round(float(relevancy_result.value), 4)
            logger.info("RAGAS answer_relevancy: %.4f", scores["answer_relevancy"])
        except Exception as exc:
            logger.warning("RAGAS answer_relevancy failed: %s", exc)
            scores["answer_relevancy"] = 0.0

    # ── MMR (Mean Reciprocal Rank) — custom implementation ───────────
    try:
        mmr = _compute_mmr(query, retrieved_contexts, llm)
        scores["mmr"] = round(mmr, 4)
        logger.info("RAGAS mmr: %.4f", scores["mmr"])
    except Exception as exc:
        logger.warning("RAGAS mmr failed: %s", exc)
        scores["mmr"] = 0.0

    return scores


def _compute_mmr(
    query: str,
    retrieved_contexts: list[str],
    llm: Any,
) -> float:
    """Compute Mean Reciprocal Rank — how early the first relevant chunk appears.

    Uses the LLM to judge each chunk's relevance to the query in order.
    The first chunk judged relevant determines the reciprocal rank:
    ``1 / rank_of_first_relevant``.  Returns 0.0 if none are relevant.

    Parameters
    ----------
    query : str
        The search query (PR diff).
    retrieved_contexts : list[str]
        Ranked list of retrieved code chunks.
    llm : LangchainLLMWrapper
        RAGAS-wrapped LLM for relevance judgments.

    Returns
    -------
    float
        MMR score in [0, 1].
    """
    if not retrieved_contexts:
        return 0.0

    # Score up to the first 5 chunks for relevance using the LLM.
    # Top-5 MMR is standard — rank 5 yields 1/5 = 0.20, and
    # anything beyond rank 5 barely changes the score anyway.
    MAX_MMR_RANK = 5
    top_chunks = retrieved_contexts[:MAX_MMR_RANK]

    for rank, chunk in enumerate(top_chunks, start=1):
        try:
            prompt = (
                f"Is the following code/document chunk relevant to the question?\n\n"
                f"Question: {query[:2_000]}\n\n"
                f"Chunk: {chunk[:2_000]}\n\n"
                f"Answer with exactly one word: YES or NO"
            )
            # Use the wrapped LangChain model for fast synchronous invoke.
            # LangchainLLMWrapper stores the LangChain model as .llm.
            raw = llm.llm.invoke(prompt)
            response_text = raw.content.strip().upper() if hasattr(raw, "content") else str(raw).strip().upper()
            if response_text.startswith("Y"):
                return 1.0 / rank
        except Exception:
            continue

    return 0.0


def log_ragas_scores_to_langfuse(
    scores: dict[str, float],
    repo: str = "",
    pr_number: int = 0,
    handler: Any = None,
) -> None:
    """Log RAGAS metrics as Langfuse scores (linked to the review trace).

    Parameters
    ----------
    scores : dict
        Output of :func:`compute_ragas_retrieval_scores`.
    repo : str
        GitHub repository name (e.g. ``\"psf/requests\"``).
    pr_number : int
        Pull request number.
    handler : Langfuse CallbackHandler or None
        Handler from the current review run, used to link scores to the trace.
    """
    try:
        from observability import log_langfuse_score
    except ImportError:
        logger.debug("observability module not available — skipping Langfuse")
        return

    for metric_name, value in scores.items():
        log_langfuse_score(
            name=f"ragas_{metric_name}",
            value=value,
            comment=f"{repo}#{pr_number} — {metric_name}",
            handler=handler,
        )


# ─── Standalone CLI ─────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: compute RAGAS scores from a JSONL dataset.

    Usage::

        python eval_data/ragas_eval.py --data eval_data/prs.jsonl
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute RAGAS retrieval quality scores on evaluation data",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to JSONL dataset (same format as evaluate.py)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write per-PR RAGAS scores (JSON)",
    )
    parser.add_argument(
        "--max-prs", type=int, default=None,
        help="Max PRs to process",
    )
    parser.add_argument(
        "--log-to-observability", action="store_true",
        help="Log RAGAS scores to Langfuse",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if not args.data:
        # If no data file specified, try default location
        default_path = os.path.join(
            os.path.dirname(__file__), "prs.jsonl"
        )
        if os.path.exists(default_path):
            args.data = default_path
        else:
            logger.error(
                "No --data provided and default not found at %s. "
                "Run eval_data/fetcher.py first.",
                default_path,
            )
            return

    with open(args.data, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.max_prs:
        entries = entries[: args.max_prs]

    all_scores: list[dict] = []
    for entry in entries:
        repo = entry.get("repo", "?")
        pr = entry.get("pr_number", 0)
        diff = entry.get("diff", "")
        ground_truth_list = entry.get("ground_truth", [])

        # Use the diff as the query, and first few human comments
        # as a proxy for "what the retrieval should have found"
        query = diff[:8_000] if diff else entry.get("title", "")
        gt_text = (
            " ".join(
                g.get("comment", "")
                for g in ground_truth_list[:5]
            )
            if ground_truth_list
            else None
        )

        # Retrieved contexts come from the graph run, but for the CLI
        # we don't re-run the graph — we just skip if no retrieval data
        # is available in the entry.
        stored_contexts = entry.get("retrieved_contexts", [])
        if not stored_contexts:
            logger.info(
                "[%s#%d] No stored retrieval contexts — skipping RAGAS",
                repo, pr,
            )
            continue

        logger.info("[%s#%d] Computing RAGAS scores …", repo, pr)
        try:
            scores = compute_ragas_retrieval_scores(
                query=query,
                retrieved_contexts=stored_contexts,
                ground_truth=gt_text,
            )
        except RuntimeError as exc:
            logger.warning("[%s#%d] %s", repo, pr, exc)
            break

        scores.update({"repo": repo, "pr_number": pr, "id": entry.get("id")})
        all_scores.append(scores)

        if args.log_to_observability:
            log_ragas_scores_to_langfuse(scores, repo=repo, pr_number=pr)

        print(
            f"  {repo}#{pr}: "
            + " | ".join(
                f"{k}={v:.3f}" for k, v in scores.items()
                if k not in ("repo", "pr_number", "id")
            )
        )

    # Print aggregate
    if all_scores:
        print()
        print("=" * 60)
        print("  Aggregate RAGAS scores")
        print("=" * 60)
        metrics = [
            k for k in all_scores[0].keys()
            if k not in ("repo", "pr_number", "id")
        ]
        for metric in metrics:
            values = [s.get(metric, 0.0) for s in all_scores]
            avg = sum(values) / len(values)
            print(f"  {metric:30s}: {avg:.4f}  (over {len(values)} PRs)")
        print()

    # Save output
    if args.output and all_scores:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2)
        logger.info("Saved RAGAS scores to %s", args.output)


if __name__ == "__main__":
    main()
