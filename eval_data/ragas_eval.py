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

from contextvars import ContextVar
import logging
import os
import threading
import time
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


# Seconds to sleep before every RAGAS LLM call. RAGAS scoring fires a
# burst of ~20-30 model calls per review (context-precision judgments,
# faithfulness verification, answer relevancy, MMR relevance checks).
# Spacing them keeps the burst under Groq/Gemini free-tier per-minute rate
# limits so scoring completes instead of tripping 429s — which previously
# caused long retry backoffs and 0.0000 scores. Override with the
# OCR_RAGAS_INTERVAL env var (float seconds, 0 disables).
_RAGAS_INTERVAL = float(os.environ.get("OCR_RAGAS_INTERVAL", "0.5") or 0.5)
_ragas_throttle_lock = threading.Lock()
_metric_deadline = ContextVar("ragas_metric_deadline", default=None)


class _ThrottledChatModel:
    """Wrap a LangChain chat model so every underlying API call RAGAS makes
    is spaced by a fixed interval.

    RAGAS's ``LangchainLLMWrapper`` (deprecated shim in 0.4.x, forwards to
    the langchain model) and the custom MMR path call the model through
    different entry points (sync/async, single/batched).  Intercepting
    ``invoke`` / ``generate`` / ``ainvoke`` / ``agenerate`` here guarantees
    every request is throttled regardless of ragas version.
    """

    def __init__(self, model, interval: float | None = None):
        self._model = model
        self._interval = _RAGAS_INTERVAL if interval is None else interval
        self._quota_exhausted = False

    def _call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403, 429} or any(value in str(exc).lower() for value in ("rate_limit_exceeded", "insufficient_quota")):
                self._quota_exhausted = True
            raise

    async def _acall(self, method, *args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403, 429} or any(value in str(exc).lower() for value in ("rate_limit_exceeded", "insufficient_quota")):
                self._quota_exhausted = True
            raise

    def _wait(self) -> None:
        deadline = _metric_deadline.get()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Metric deadline exceeded; no further judge calls")
        if self._quota_exhausted:
            raise RuntimeError("Evaluation stopped: provider authentication or quota limit; no further calls will be made for this evaluation.")
        if self._interval > 0:
            with _ragas_throttle_lock:
                time.sleep(self._interval)

    # ── Sync entry points ───────────────────────────────────────────────
    def invoke(self, *args, **kwargs):
        self._wait()
        return self._call(self._model.invoke, *args, **kwargs)

    def generate(self, messages, *args, **kwargs):
        # Serialize LangChain batch calls into individual throttled
        # requests so every underlying API call is spaced by the interval.
        if (
            isinstance(messages, list)
            and messages
            and isinstance(messages[0], list)
        ):
            from langchain_core.outputs import Generation, LLMResult

            generations = []
            for msgs in messages:
                self._wait()
                # invoke() does not accept callbacks/run_manager/config that
                # ragas may forward to generate() — drop them here.
                resp = self._call(self._model.invoke, msgs)
                text = resp.content if hasattr(resp, "content") else str(resp)
                generations.append([Generation(text=text, message=resp)])
            return LLMResult(generations=generations, llm_output=None)
        self._wait()
        return self._call(self._model.generate, messages, *args, **kwargs)

    # ── Async entry points ──────────────────────────────────────────────
    async def ainvoke(self, *args, **kwargs):
        self._wait()
        return await self._acall(self._model.ainvoke, *args, **kwargs)

    async def agenerate(self, messages, *args, **kwargs):
        if (
            isinstance(messages, list)
            and messages
            and isinstance(messages[0], list)
        ):
            from langchain_core.outputs import Generation, LLMResult

            generations = []
            for msgs in messages:
                self._wait()
                # Same kwargs caveat as generate(): ainvoke has no callbacks.
                resp = await self._acall(self._model.ainvoke, msgs)
                text = resp.content if hasattr(resp, "content") else str(resp)
                generations.append([Generation(text=text, message=resp)])
            return LLMResult(generations=generations, llm_output=None)
        self._wait()
        return await self._acall(self._model.agenerate, messages, *args, **kwargs)

    # ── Everything else passes through ──────────────────────────────────
    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _wrap_evaluator(llm):
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    return LangchainLLMWrapper(_ThrottledChatModel(llm), run_config=RunConfig(timeout=12, max_retries=1, max_wait=0, max_workers=1))


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

            llm = create_llm(endpoint=endpoint, temperature=0, timeout=15, max_retries=0)
            logger.info("RAGAS evaluator: using custom endpoint %s", endpoint)
            return _wrap_evaluator(llm)

        from llm_factory import _get_groq_chat, _get_gemini_chat

        # Keep evaluation bounded: RAGAS runs several provider calls and
        # retrying each one can otherwise hold a review open for minutes.
        configured_model = os.getenv("RAGAS_MODEL", "").strip()
        if configured_model.startswith("groq/"):
            configured_model = configured_model.split("/", 1)[1]
        groq_kwargs = {"temperature": 0, "max_retries": 0, "timeout": 15}
        if configured_model:
            groq_kwargs["model"] = configured_model
        llm = _get_groq_chat(**groq_kwargs)
        if llm is None:
            logger.info(
                "RAGAS evaluator: GROQ_API_KEY not set — falling back to Gemini"
            )
            llm = _get_gemini_chat(temperature=0, max_retries=0, timeout=15)
        if llm is None:
            raise ValueError(
                "No LLM available for RAGAS scoring. "
                "Set GROQ_API_KEY (preferred) or GEMINI_API_KEY."
            )
        logger.info(
            "RAGAS evaluator: using %s (throttled %ss)",
            type(llm).__name__,
            _RAGAS_INTERVAL,
        )
        return _wrap_evaluator(llm)
    except ValueError as exc:
        logger.warning("RAGAS evaluator: %s", exc)
        raise RuntimeError(
            "No LLM available for RAGAS scoring. "
            "Set GROQ_API_KEY (preferred) or GEMINI_API_KEY."
        ) from exc


# ─── Score computation ─────────────────────────────────────────────────────


def _score_one_metric(name, sample, llm, timeout):
    from ragas import metrics
    from ragas.run_config import RunConfig
    classes = {"context_precision": "LLMContextPrecisionWithoutReference", "faithfulness": "Faithfulness", "context_recall": "ContextRecall"}
    scorer = getattr(metrics, classes[name])(llm=llm)
    scorer.init(RunConfig(timeout=timeout, max_retries=1, max_wait=0, max_workers=1))
    return float(scorer.single_turn_score(sample))


def compute_ragas_report(query, retrieved_contexts, ground_truth=None, answer=None, endpoint=None, *, metric_timeout=None, progress_callback=None):
    """Run independent bounded RAGAS metrics, preserving successes and real failures.

    No local embedding model is loaded. Lexical overlap is never substituted for
    RAGAS. Input caps reduce judge latency and are disclosed in the report.
    """
    import math
    import queue
    from contextvars import copy_context
    started = time.monotonic()
    total_budget = max(1.0, float(os.getenv("OCR_RAGAS_TIMEOUT_SECONDS", "30")))
    per_metric = max(.05, float(metric_timeout or os.getenv("OCR_RAGAS_METRIC_TIMEOUT_SECONDS", "12")))
    names = ["context_precision", "faithfulness"]
    if ground_truth:
        names.append("context_recall")
    report = {"scores": {}, "metrics": {}, "evaluation": {"engine": "RAGAS", "local_models": False, "metric_timeout_seconds": per_metric, "input_truncated": len(query) > 6000 or len(answer or "") > 4000 or len(retrieved_contexts) > 2 or any(len(c) > 4000 for c in retrieved_contexts)}}
    def finish(name, status, value=None, error=None, duration=0):
        report["metrics"][name] = {"status": status, "value": value, "error": error, "duration_seconds": round(duration, 3)}
        if value is not None:
            report["scores"][name] = value
        if progress_callback:
            progress_callback(report)
    for name, reason in [("answer_relevancy", "Requires embeddings; local models are disabled"), ("mmr", "Extra judge calls disabled in hosted evaluation")]:
        finish(name, "skipped", error=reason)
    if not ground_truth:
        finish("context_recall", "skipped", error="No reference answer provided")
    if not answer or not retrieved_contexts:
        for name in names:
            finish(name, "skipped", error="A complete answer and context are required")
        return report
    try:
        if not _check_ragas():
            raise RuntimeError("RAGAS is not installed")
        from ragas.dataset_schema import SingleTurnSample
        llm = _get_evaluator_llm(endpoint=endpoint)
        sample = SingleTurnSample(user_input=query[:6000], response=answer[:4000], retrieved_contexts=[c[:4000] for c in retrieved_contexts[:2]], reference=ground_truth[:4000] if ground_truth else None)
    except Exception as exc:
        for name in names:
            finish(name, "failed", error=type(exc).__name__)
        return report
    pending = {}
    for name in names:
        remaining = total_budget - (time.monotonic() - started) - 1
        if remaining <= 0:
            finish(name, "timed_out", error="Evaluation setup exhausted deadline")
            continue
        deadline = time.monotonic() + min(per_metric, remaining)
        output = queue.Queue(maxsize=1)
        def run(metric=name, target=output, limit=min(per_metric, remaining)):
            metric_started = time.monotonic()
            token = _metric_deadline.set(metric_started + limit)
            try:
                value = _score_one_metric(metric, sample, llm, limit)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid metric value")
                target.put(("completed", round(value, 4), None, time.monotonic() - metric_started))
            except Exception as exc:
                target.put(("failed", None, type(exc).__name__, time.monotonic() - metric_started))
            finally:
                _metric_deadline.reset(token)
        threading.Thread(target=copy_context().run, args=(run,), daemon=True).start()
        pending[name] = (output, deadline)
    while pending:
        for name, (output, deadline) in list(pending.items()):
            try:
                status, value, error, duration = output.get_nowait()
            except queue.Empty:
                if time.monotonic() < deadline:
                    continue
                status, value, error, duration = "timed_out", None, "Metric deadline exceeded", per_metric
            finish(name, status, value, error, duration)
            del pending[name]
        if pending:
            time.sleep(.01)
    report["evaluation"]["duration_seconds"] = round(time.monotonic() - started, 3)
    return report


def compute_ragas_retrieval_scores(query, retrieved_contexts, ground_truth=None, answer=None, endpoint=None):
    """Compatibility mapping. Failed/skipped metrics remain None, never invented scores."""
    report = compute_ragas_report(query, retrieved_contexts, ground_truth, answer, endpoint)
    return {name: metric["value"] for name, metric in report["metrics"].items()}


def _create_ragas_embeddings():
    """Return a lightweight embedding object compatible with RAGAS metrics.

    Wraps the ChromaDB ONNX MiniLM embedding function so that metrics like
    ``ResponseRelevancy`` can compute cosine similarity between embeddings.
    The underlying ONNX model is cached globally and loads in <1 s.
    """
    if os.getenv("OPENCODEREVIEW_ALLOW_LOCAL_EMBEDDINGS", "false").lower() not in {"1", "true", "yes"}:
        logger.info("Local RAGAS embeddings disabled; skipping answer relevancy")
        return None
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
    MAX_MMR_RANK = 2
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
            # Space the relevance judgments too — this unwrapped path bypasses
            # the model wrapper on some ragas versions, so sleep explicitly
            # (unless the wrapper is already in the path, to avoid double-sleep).
            if not isinstance(langchain_model, _ThrottledChatModel):
                time.sleep(_RAGAS_INTERVAL)
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
