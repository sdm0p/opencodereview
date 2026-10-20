#!/usr/bin/env python3
"""Fetch a bounded, diff-backed subset of Code Review Bench.

This only downloads public PR diffs and keeps the expert comments as labels.
It never posts to GitHub or mutates repositories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("eval_data/diff_backed_fixture.jsonl"))
    args = parser.parse_args()

    source = hf_hub_download(
        "code-review-bench/code-review-bench",
        filename="offline_golden_issues.parquet",
        repo_type="dataset",
    )
    rows = pq.read_table(source).to_pylist()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["pr_url"], []).append(row)

    output: list[dict] = []
    for url, labels in list(grouped.items()):
        if len(output) >= args.limit:
            break
        try:
            response = requests.get(f"{url}.diff", timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"skip {url}: {exc}")
            continue
        if not response.text.strip():
            continue
        first = labels[0]
        output.append(
            {
                "task_id": url,
                "pr_url": url,
                "source_repo": first["source_repo"],
                "pr_title": first["pr_title"],
                "diff": response.text,
                "gold_issues": labels,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output),
        encoding="utf-8",
    )
    print(f"wrote {len(output)} fixtures to {args.output}")


if __name__ == "__main__":
    main()
