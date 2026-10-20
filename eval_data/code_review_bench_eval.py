#!/usr/bin/env python3
"""Score OpenCodeReview predictions against Code Review Bench offline rows.

Example:
  python eval_data/code_review_bench_eval.py --predictions predictions.jsonl --max-rows 50
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

TOLERANCE = 3
STOP = {"the","a","an","and","or","to","of","in","is","are","this","that","for","with","on","it","as","be","by"}

def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", (text or "").lower()) if w not in STOP and len(w) > 2}

def issue_fields(row: dict) -> list[dict]:
    raw = row.get("issues") or row.get("gold_issues") or row.get("review_comments") or row.get("comments")
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if row.get("comment") or row.get("issue"):
        return [{"comment": row.get("comment") or row.get("issue"), "file_path": row.get("file_path") or row.get("path"), "line": row.get("line") or row.get("line_number", 0), "severity": row.get("severity")}]
    return []

def match(pred: dict, gold: dict) -> tuple[float, bool]:
    pf, gf = pred.get("file_path") or pred.get("path"), gold.get("file_path") or gold.get("path")
    if pf and gf and pf != gf:
        return 0.0, False
    pl = pred.get("line_start", pred.get("line", 0)) or 0
    gl = gold.get("line_start", gold.get("line", 0)) or 0
    if gl and pl and abs(int(pl) - int(gl)) > TOLERANCE:
        return 0.0, False
    overlap = len(words(pred.get("comment", "")) & words(gold.get("comment") or gold.get("body") or gold.get("issue_text", "")))
    denom = max(1, len(words(gold.get("comment") or gold.get("body") or gold.get("issue_text", ""))))
    score = min(1.0, overlap / denom)
    return score, score >= 0.2 or (not gold.get("comment") and pf == gf)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("eval_data/code_review_bench_results.json"))
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--fixture", type=Path, help="Optional diff-backed JSONL fixture with gold_issues per PR")
    args = ap.parse_args()
    preds = {str(x.get("task_id") or x.get("id") or f"{x.get('pr_url')}#{x.get('issue_index', 0)}" or f"{x.get('repo')}#{x.get('pr_number')}"): x for x in (json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip())}
    if args.fixture:
        rows = [json.loads(line) for line in args.fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub and pyarrow first") from exc
        offline_path = hf_hub_download("code-review-bench/code-review-bench", filename="offline_golden_issues.parquet", repo_type="dataset")
        rows = pq.read_table(offline_path).to_pylist()
    totals = {"rows": 0, "gold_findings": 0, "pred_findings": 0, "matched": 0, "false_positives": 0, "missed": 0, "severity_correct": 0, "category_correct": 0}
    failures = []
    for row in rows:
        if args.max_rows and totals["rows"] >= args.max_rows: break
        key = str(row.get("task_id") or row.get("id") or f"{row.get('pr_url')}#{row.get('issue_index', 0)}" or f"{row.get('repo')}#{row.get('pr_number')}")
        pred = preds.get(key, {"findings": []})
        gold = row.get("gold_issues") if args.fixture else issue_fields(row)
        gold = gold or []
        generated = pred.get("findings") or []
        used = set(); matched = 0
        for finding in generated:
            best = None
            for i, target in enumerate(gold):
                if i in used: continue
                score, ok = match(finding, target)
                if ok and (best is None or score > best[0]): best = (score, i)
            if best:
                used.add(best[1]); matched += 1
                target = gold[best[1]]
                if finding.get("severity") and target.get("severity") and str(finding["severity"]).lower() == str(target["severity"]).lower():
                    totals["severity_correct"] += 1
                if finding.get("category") and target.get("category") and str(finding["category"]).lower() == str(target["category"]).lower():
                    totals["category_correct"] += 1
        totals["rows"] += 1; totals["gold_findings"] += len(gold); totals["pred_findings"] += len(generated); totals["matched"] += matched
        totals["false_positives"] += len(generated) - matched; totals["missed"] += len(gold) - matched
        if len(generated) - matched or len(gold) - matched:
            failures.append({"task_id": key, "gold": len(gold), "predicted": len(generated), "matched": matched})
    precision = totals["matched"] / totals["pred_findings"] if totals["pred_findings"] else 0.0
    recall = totals["matched"] / totals["gold_findings"] if totals["gold_findings"] else 0.0
    result = {**totals, "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
              "severity_accuracy": totals["severity_correct"] / totals["matched"] if totals["matched"] else 0.0,
              "category_accuracy": totals["category_correct"] / totals["matched"] if totals["matched"] else 0.0,
              "false_positive_rate": totals["false_positives"] / totals["pred_findings"] if totals["pred_findings"] else 0.0,
              "failures": failures}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rows","precision","recall","f1","false_positives","missed")}, indent=2))

if __name__ == "__main__":
    main()

