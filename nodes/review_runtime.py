"""Bound reviewer latency and validate provider responses without hiding failures."""
from __future__ import annotations

from contextvars import copy_context
import json
import os
import queue
import re
import threading
import time

from pydantic import ValidationError


def changed_context(changed_file, mode):
    """Fast mode includes numbered windows around changes, not repeated whole files."""
    content = changed_file.content or ''
    if mode != 'fast' or not changed_file.diff_hunk:
        return content
    lines = content.splitlines()
    selected = set()
    for start, count in re.findall(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', changed_file.diff_hunk):
        first = max(0, int(start) - 1)
        selected.update(range(max(0, first - 12), min(len(lines), first + int(count or 1) + 12)))
    return '\n'.join(f'{index + 1}: {lines[index]}' for index in sorted(selected)) if selected else content


def invoke_review(llm, schema, messages, mode="thorough"):
    """One request plus one format repair, within a total wall-clock budget.

    A daemon worker makes the caller deadline effective even with a provider
    SDK that ignores its native request timeout. Native timeouts still close
    underlying connections; timed-out workers never modify graph state.
    """
    budget = float(os.getenv("OCR_REVIEWER_TIMEOUT_SECONDS", "90"))
    if mode == "fast":
        budget = min(budget, 45.0)
    deadline = time.monotonic() + budget
    instructions = (
        "Return ONLY a JSON object matching this schema, never a bare list. "
        "If there are no findings, return {\"findings\": []}. Schema: "
        + json.dumps(schema.model_json_schema())
    )
    messages = [*messages, {"role": "user", "content": instructions}]
    structured = llm.with_structured_output(schema)
    for attempt in range(2):
        result_queue = queue.Queue(maxsize=1)
        def run(target=result_queue, request=list(messages)):
            try:
                target.put((True, structured.invoke(request)))
            except Exception as exc:
                target.put((False, exc))
        threading.Thread(target=copy_context().run, args=(run,), daemon=True).start()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Reviewer exceeded {budget:g}s deadline")
        try:
            ok, result = result_queue.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError(f"Reviewer exceeded {budget:g}s deadline") from None
        if ok:
            try:
                if isinstance(result, schema):
                    return result
                return schema.model_validate(result)
            except ValidationError as exc:
                result = exc
        # No retries for auth, rate limits, transport errors or timeouts.
        if attempt or not (isinstance(result, ValidationError) or
                           type(result).__name__ == "OutputParserException"):
            raise result
        messages.append({"role": "user", "content":
            "The previous response did not match the schema. Return the required JSON object with a findings array."})


def stage_result(reviewer, started, *, error=None, llm=None):
    """Safe stage metadata; exception text may contain secrets, so omit it."""
    return {"reviewer": reviewer, "status": "failed" if error else "completed",
            "error": type(error).__name__ if error else None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "model": str(getattr(llm, "model_name", None) or getattr(llm, "model", "")) if llm else ""}
