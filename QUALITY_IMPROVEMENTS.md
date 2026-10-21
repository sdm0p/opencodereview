# Review quality improvements

Implemented changes:
- Group test suggestions separately, cap low/info, and exclude them from the risk score.
- Check changed-file/line overlap and derive excerpts from actual source. Source grounding does not prove semantic correctness.
- Compiler claims without diagnostics are flagged for verification and excluded from the score.
- Explain risk-score deductions and disclose prompt truncation, reviewed files, Fast windows, incremental scope, and reviewer status.
- Withhold ungrounded findings from GitHub posting and report the withheld count.
- Record owner-scoped correct/incorrect/useful feedback on completed findings in the job database.
- Edit/delete/default provider controls, with browser-local persistence and last successful connection/JSON test metadata.
- RAGAS metrics have individual deadlines, independent result statuses, explicit skipped unsupported metrics, and bounded hosted inputs. No lexical fallback is labeled RAGAS.
- Preserve per-metric successes/failures and evaluation configuration through polling and database persistence. Diff-proxy context is disclosed.
- Distinguish provider-reported cost, estimates, free-model estimates, and unknown costs. Response latency is not mislabeled TTFT.
- Offline regression tool compares stored predictions against located labels with one-to-one matching; partial gold labels do not yield precision claims.

Run regression matching without inference:

```
python -m eval_data.offline_regression --predictions review.json --fixture eval_data/curated_located_fixture.jsonl --output eval_data/regression_report.json
```

The curated initial fixture is partial. Location/category/concept matches are evaluation proxies, not proof that a finding is correct. No local models are loaded.

Validation: backend and frontend regression tests plus the production frontend build. Live verification is reported separately; no improvement percentages are claimed without a controlled benchmark.

Live verification (2026-10-21): Vercel UI and HF backend; calcom/cal.diy PR #10600 with the saved MiniMax free endpoint in Fast mode completed in 40 seconds. All three reviewers completed. Results grouped 9 test suggestions separately from 3 low-severity findings; score breakdown excluded the suggestions. Feedback saved successfully. Provider connection test returned valid JSON in 1.2 seconds; default selection survived refresh. RAGAS context precision completed (1.000, using a truncated PR-diff proxy); faithfulness returned a provider connection error and the UI reported partial evaluation. This run is workflow evidence, not proof of finding correctness or retrieval accuracy.

Follow-up live verification after isolated evaluator clients: review f0fa11f7 completed in 53 seconds with all reviewers completed, 1 low bug, 1 medium security concern, 7 test suggestions, and a 9.5/10 risk estimate. Context precision completed (0.000); faithfulness reached its 12-second deadline and remained unscored. Differences between runs reinforce that this hosted judge/model is variable; these proxy scores are not correctness guarantees.
