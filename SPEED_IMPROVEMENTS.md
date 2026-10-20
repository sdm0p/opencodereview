# Review speed improvements

Implemented in this revision:
- Fast/Thorough modes: reviewer wall deadlines 45/90 seconds, native call timeouts 40/60 seconds; total pipeline also includes ingestion and aggregation.
- SDK retries disabled; one format repair within the reviewer deadline. No retries for auth/quota/transport failures.
- Failed reviewers make the result incomplete, with no numeric score and no posting.
- Background RAGAS uses a captured selected endpoint, stops after quota/auth failures, and is never automatically replayed after restart.
- Provider probe records real request latency and tests JSON output shape.
- Fast mode supplies numbered code windows around changes; Thorough retains existing full context bounds.
- Conservative routing: Fast documentation-only changes skip test coverage explicitly. Correctness/security remain; code, configuration, and unknown changes retain all specialists.
- Token budgets 1200/2000 per reviewer in Fast/Thorough. Model-specific reasoning parameters are deliberately not forced on incompatible providers.
- Polling streams completed stages, partial findings, reviewer statuses, timings, queue position, and background evaluation status.
- Cooperative cancellation discards late results; native network calls finish within their own timeout.
- Active duplicate prevention and 15-minute preview reuse require matching owner/configuration/prompt+node hashes and fresh GitHub head/base commits. Approval rechecks commits.
- Explicit incremental baseline comparisons reject divergent/truncated comparisons and are preview-only.
- GitHub file fetching runs concurrently with a maximum of four workers.
- User-selected saved providers remain the explicit alternative when a provider fails; no silent use of another user's/operator's key.

Validation: backend and frontend regression tests use fake providers; no local model runs. Live deployment verification is reported separately. No percentage speed improvement is claimed before measuring the same PR/model/configuration.

Known limits: cancellation is cooperative; browser-saved credentials remain local to the browser; cross-instance job dispatch requires a durable queue. Detailed provider-specific reasoning controls and automatic cross-provider failover are not implemented.
