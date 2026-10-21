"""Offline location/category/concept matching, with no provider or model calls.

Usage: python -m eval_data.offline_regression --predictions review.json --fixture labelled.jsonl --output report.json
Gold rows require repo/pr_number (or pr_url) and findings/gold_issues with
file_path, line_start/line_end, category, comment, and optional concept_terms.
Scores are deterministic matching proxies, not proof of semantic correctness.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def identity(row):
    url = row.get("pr_url") or row.get("task_id")
    if url:
        return str(url).replace("https://github.com/", "").replace("/pull/", "#")
    return f"{row.get('repo')}#{row.get('pr_number')}"


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
        return rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def match_score(prediction, gold):
    if prediction.get("file_path") != gold.get("file_path") or not gold.get("file_path"):
        return None
    if prediction.get("category") != gold.get("category") or not gold.get("category"):
        return None
    ps = int(prediction.get("line_start") or 0)
    gs = int(gold.get("line_start") or gold.get("line") or 0)
    if not ps or not gs:
        return None
    pe, ge = int(prediction.get("line_end") or ps), int(gold.get("line_end") or gs)
    if ps > ge + 3 or gs > pe + 3:
        return None
    words = set(re.findall(r"[a-z]+", prediction.get("comment", "").lower()))
    concepts = gold.get("concept_terms") or re.findall(r"[a-z]{4,}", gold.get("comment", "").lower())
    overlap = sum(1 for term in concepts if term.lower() in words)
    return overlap / max(1, len(concepts)) if overlap else None


def evaluate(predictions, fixtures):
    indexed = {identity(row): row for row in fixtures}
    partial_labels = False
    total = dict(matched=0, predicted_bugs=0, gold_bugs=0, suggestions=0, duplicate_predictions=0, unlocatable_gold=0, rows=0)
    cases = []
    for prediction in predictions:
        key = identity(prediction)
        if key not in indexed:
            cases.append({"id": key, "status": "skipped", "reason": "No labelled fixture"})
            continue
        total["rows"] += 1
        partial_labels = partial_labels or indexed[key].get("label_scope") != "exhaustive"
        gold = indexed[key].get("findings") or indexed[key].get("gold_issues") or []
        total["unlocatable_gold"] += sum(not item.get("file_path") or not (item.get("line_start") or item.get("line")) for item in gold)
        valid = [item for item in gold if item.get("file_path") and (item.get("line_start") or item.get("line"))]
        bugs = [item for item in prediction.get("findings", []) if item.get("category") != "test_coverage" and item.get("kind") not in {"suggestion", "test_suggestion"}]
        total["suggestions"] += len(prediction.get("findings", [])) - len(bugs)
        total["predicted_bugs"] += len(bugs)
        total["gold_bugs"] += len(valid)
        edges = sorted([(score, pi, gi) for pi, pred in enumerate(bugs) for gi, item in enumerate(valid) if (score := match_score(pred, item)) is not None], reverse=True)
        used_predictions, used_gold = set(), set()
        for score, pi, gi in edges:
            if pi not in used_predictions and gi not in used_gold:
                used_predictions.add(pi)
                used_gold.add(gi)
        total["matched"] += len(used_gold)
        duplicate = sum(any(match_score(pred, item) is not None for item in valid) for pi, pred in enumerate(bugs) if pi not in used_predictions)
        total["duplicate_predictions"] += duplicate
        cases.append({"id": key, "status": prediction.get("status"), "matched": len(used_gold), "false_positives": len(bugs) - len(used_predictions), "missed": len(valid) - len(used_gold), "elapsed_seconds": prediction.get("elapsed_seconds"), "reviewer_failures": sum(item.get("status") == "failed" for item in prediction.get("review_stages", []))})
    return {"method": "location_category_concept_proxy_v1", "limitations": "Deterministic matching is not a semantic correctness judge. Labels and unmatched/duplicate predictions require human review. Unlocatable gold is excluded, never counted as a pass.", "totals": total, "label_scope": "partial" if partial_labels else "exhaustive", "precision": total["matched"] / total["predicted_bugs"] if total["predicted_bugs"] and not partial_labels and total["gold_bugs"] else None, "annotated_match_rate": total["matched"] / total["predicted_bugs"] if total["predicted_bugs"] and total["gold_bugs"] else None, "recall": total["matched"] / total["gold_bugs"] if total["gold_bugs"] else None, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", default="eval_data/regression_report.json")
    args = parser.parse_args()
    report = evaluate(load_rows(args.predictions), load_rows(args.fixture))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
