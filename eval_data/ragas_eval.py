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
    """Return True if RAGAS is available.

    Note: ``litellm`` is NOT required because RAGAS scoring uses the
    project's own LLM factory (Gemini or Groq via LangChain), not
    litellm internally.
    """
    global _RAGAS_AVAILABLE
    if _RAGAS_AVAILABLE is not None:
        return _RAGAS_AVAILABLE
    try:
        import ragas  # noqa: F401
        _RAGAS_AVAILABLE = True
    except ImportError:
        logger.warning(
            "ragas not installed. Install with: pip install ragas"
        )
        _RAGAS_AVAILABLE = False
    return _RAGAS_AVAILABLE


def _get_evaluator_llm(endpoint: Optional[str] = None):
    """Return a RAGAS-compatible LLM, reusing the project's LLM factory.

    Priority:
    1. **Custom endpoint** — when *endpoint* names a configured
       ``OCR_ENDPOINT_*`` config, it is used (any provider the user added).
    2. **Groq** (``llama-3.3-70b-versatile``) via ``GROQ_API_KEY``
    3. **Google Gemini** (``gemini-3.1-flash-lite``) via ``GEMINI_API_KEY``

    Groq is preferred over Gemini because scoring fires many LLM calls per
    review and the Gemini free tier (15 req/min) is already shared with the
    main review pipeline — using Groq's separate quota pool avoids the 429
    rate-limit failures that used to zero out the metrics.

    RAGAS v0.3+ uses ``LangchainLLMWrapper`` to wrap LangChain models.
    """
    from ragas.llms import LangchainLLMWrapper

    try:
        if endpoint:
            from llm_factory import create_llm

            llm = create_llm(endpoint=endpoint, temperature=0)
            logger.info("RAGAS evaluator: using custom endpoint %s", endpoint)
            return LangchainLLMWrapper(llm)

        from llm_factory import _get_groq_chat, _get_gemini_chat

        llm = _get_groq_chat(temperature=0)
        if llm is None:
            logger.info(
                "RAGAS evaluator: GROQ_API_KEY not set — falling back to Gemini"
            )
            llm = _get_gemini_chat(temperature=0)
        if llm is None:
            raise ValueError(
                "No LLM available for RAGAS scoring. "
                "Set GROQ_API_KEY (preferred) or GEMINI_API_KEY."
            )
        logger.info(
            "RAGAS evaluator: using %s",
            type(llm).__name__,
        )
        return LangchainLLMWrapper(llm)
    except ValueError as exc:
        logger.warning("RAGAS evaluator: %s", exc)
        raise RuntimeError(
            "No LLM available for RAGAS scoring. "
            "Set GROQ_API_KEY (preferred) or GEMINI_API_KEY."
        ) from exc


# ─── Score computation ─────────────────────────────────────────────────────


def compute_ragas_retrieval_scores(
    query: str,
    retrieved_contexts: list[str],
    ground_truth: Optional[str] = None,
    answer: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> dict[str, float | None]:
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
    endpoint : str or None
        Optional name of a configured custom LLM endpoint
        (``OCR_ENDPOINT_*``) to score with.  ``None`` uses the built-in
        Groq → Gemini priority.

    Returns
    -------
    dict
        Mapping of metric name → score (0–1 range, higher is better),
        or ``None`` when that metric's computation failed (e.g. LLM
        rate limit / error).  Possible keys: ``context_precision``,
        ``context_recall``, ``faithfulness``, ``answer_relevancy``,
        ``mmr``.

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

    llm = _get_evaluator_llm(endpoint=endpoint)
    scores: dict[str, float | None] = {}

    # ── Context Precision (reference-free) ───────────────────────────
    # Uses LLMContextPrecisionWithoutReference which needs user_input +
    # response + retrieved_contexts (no ground-truth reference required).
    if answer:
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference

            scorer = LLMContextPrecisionWithoutReference(llm=llm)
            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=retrieved_contexts,
            )
            result = scorer.single_turn_score(sample)
            scores["context_precision"] = round(float(result), 4)
            logger.info("RAGAS context_precision: %.4f", scores["context_precision"])
        except Exception as exc:
            logger.warning("RAGAS context_precision failed: %s", exc)
            scores["context_precision"] = None

    # ── Context Recall (requires ground_truth) ───────────────────────
    if ground_truth:
        try:
            from ragas.metrics import ContextRecall

            recall_scorer = ContextRecall(llm=llm)
            recall_sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
            recall_result = recall_scorer.single_turn_score(recall_sample)
            scores["context_recall"] = round(float(recall_result), 4)
            logger.info("RAGAS context_recall: %.4f", scores["context_recall"])
        except Exception as exc:
            logger.warning("RAGAS context_recall failed: %s", exc)
            scores["context_recall"] = None

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
            faithful_result = faithfulness_scorer.single_turn_score(faithful_sample)
            scores["faithfulness"] = round(float(faithful_result), 4)
            logger.info("RAGAS faithfulness: %.4f", scores["faithfulness"])
        except Exception as exc:
            logger.warning("RAGAS faithfulness failed: %s", exc)
            scores["faithfulness"] = None

    # ── Answer Relevancy (ResponseRelevancy) ─────────────────────────
    if answer:
        try:
            from ragas.metrics import ResponseRelevancy

            # ResponseRelevancy also needs an embedding model for cosine
            # similarity.  We create a lightweight one from the project's
            # ChromaDB embedding function (ONNX MiniLM, already cached).
            embeddings = _create_ragas_embeddings()
            if embeddings is None:
                raise RuntimeError(
                    "No embedding model available for ResponseRelevancy — "
                    "install chromadb"
                )

            relevancy_scorer = ResponseRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=1,  # Groq only supports n=1
            )
            relevancy_sample = SingleTurnSample(
                user_input=query,
                response=answer,
            )
            relevancy_result = relevancy_scorer.single_turn_score(relevancy_sample)
            scores["answer_relevancy"] = round(float(relevancy_result), 4)
            logger.info("RAGAS answer_relevancy: %.4f", scores["answer_relevancy"])
        except Exception as exc:
            logger.warning("RAGAS answer_relevancy failed: %s", exc)
            scores["answer_relevancy"] = None

    # ── MMR (Mean Reciprocal Rank) — custom implementation ───────────
    try:
        mmr = _compute_mmr(query, retrieved_contexts, llm)
        scores["mmr"] = round(mmr, 4)
        logger.info("RAGAS mmr: %.4f", scores["mmr"])
    except Exception as exc:
        logger.warning("RAGAS mmr failed: %s", exc)
        scores["mmr"] = None

    # ── Context Precision (reference-based, fallback when no answer) ─
    if "context_precision" not in scores and ground_truth:
        try:
            from ragas.metrics import ContextPrecision

            scorer = ContextPrecision(llm=llm)
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
            result = scorer.single_turn_score(sample)
            scores["context_precision"] = round(float(result), 4)
            logger.info("RAGAS context_precision: %.4f", scores["context_precision"])
        except Exception as exc:
            logger.warning("RAGAS context_precision failed: %s", exc)
            scores["context_precision"] = None

    return scores


def _create_ragas_embeddings():
    """Return a lightweight embedding object compatible with RAGAS metrics.

    Wraps the ChromaDB ONNX MiniLM embedding function so that metrics like
    ``ResponseRelevancy`` can compute cosine similarity between embeddings.
    The underlying ONNX model is cached globally and loads in <1 s.
    """
    try:
        from chromadb.utils import embedding_functions

        if hasattr(embedding_functions, "ONNXMiniLM_L6_V2"):
            ef = embedding_functions.ONNXMiniLM_L6_V2()
        else:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

        class _RagasEmbeddings:
            def __init__(self, ef):
                self._ef = ef

            def embed_query(self, text: str) -> list[float]:
                result = self._ef([text])[0]
                return list(result) if not isinstance(result, list) else result

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                results = self._ef(texts)
                return [list(r) if not isinstance(r, list) else r for r in results]

        return _RagasEmbeddings(ef)
    except Exception as exc:
        logger.warning("Cannot create RAGAS embeddings: %s", exc)
        return None


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
        MMR score in [0, 1]; 0.0 when chunks were judged but none were
        relevant.

    Raises
    ------
    RuntimeError
        If every relevance judgment failed (e.g. LLM rate-limited) —
        callers treat this as a failed metric rather than a fake 0.0.
    """
    if not retrieved_contexts:
        return 0.0

    # Score up to the first 5 chunks for relevance using the LLM.
    # Top-5 MMR is standard — rank 5 yields 1/5 = 0.20, and
    # anything beyond rank 5 barely changes the score anyway.
    MAX_MMR_RANK = 5
    top_chunks = retrieved_contexts[:MAX_MMR_RANK]

    judged = 0
    for rank, chunk in enumerate(top_chunks, start=1):
        try:
            prompt = (
                f"Is the following code/document chunk relevant to the question?\n\n"
                f"Question: {query[:2_000]}\n\n"
                f"Chunk: {chunk[:2_000]}\n\n"
                f"Answer with exactly one word: YES or NO"
            )
            # Use the wrapped LangChain model for fast synchronous invoke.
            # LangchainLLMWrapper stores the model as .langchain_llm in ragas
            # 0.4.x and as .llm in 0.3.x — accept both.
            langchain_model = getattr(llm, "langchain_llm", None) or getattr(llm, "llm", None)
            raw = langchain_model.invoke(prompt)
            response_text = raw.content.strip().upper() if hasattr(raw, "content") else str(raw).strip().upper()
            if response_text.startswith("Y"):
                return 1.0 / rank
            judged += 1  # judge responded — chunk deemed not relevant
        except Exception:
            continue

    if judged == 0:
        # Every relevance judgment failed (e.g. LLM rate-limited) — surface
        # this as a failed metric instead of a misleading 0.0.
        raise RuntimeError("all MMR relevance judgments failed")

    return 0.0


def log_ragas_scores_to_langfuse(
    scores: dict[str, float | None],
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
        if value is None:
            continue
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
                if k not in ("repo", "pr_number", "id") and v is not None
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
            values = [
                s.get(metric) for s in all_scores
                if s.get(metric) is not None
            ]
            if not values:
                print(f"  {metric:30s}: n/a  (no successful scores)")
                continue
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
