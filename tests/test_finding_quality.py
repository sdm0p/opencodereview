from unittest.mock import Mock

from nodes.aggregator import aggregator_node, _deduplicate
from nodes.finding_quality import calibrate_findings, review_coverage
from nodes.executor import executor_node, _finding_body
from state import ChangedFile, Finding, OpenCodeReviewState, Severity, Verdict


def state_for(*findings):
    return OpenCodeReviewState(repo='owner/repo', pr_number=1, diff='diff', changed_files=[ChangedFile(
        path='auth.ts', status='modified', content='const codes = read();\nconsume(codes);\nreturn true;',
        diff_hunk='@@ -1,2 +1,3 @@\n const codes = read();\n+consume(codes);\n return true;')], findings=list(findings))


def finding(**changes):
    return Finding(file_path='auth.ts', line_start=2, line_end=2, severity=Severity.HIGH,
                   category='correctness', comment='Concurrent calls consume the same code.', confidence=.98, **changes)


def test_test_suggestions_cannot_drive_risk_score_to_zero():
    raw = finding()
    raw.category = 'test_coverage'
    raw.comment = 'The supplied test context does not exercise this branch.'
    result = aggregator_node(state_for(raw))
    calibrated = result['final_findings'][0]
    assert calibrated.kind == 'test_suggestion'
    assert calibrated.severity == Severity.LOW
    assert result['verdict'].overall_score == 10
    assert result['verdict'].high_count == 0
    assert result['verdict'].score_breakdown['test_suggestions_excluded'] == 1
    assert result['verdict'].recommendation == 'comment'


def test_evidence_is_taken_from_actual_source_not_model_quote():
    raw = finding(evidence_excerpt='invented code')
    calibrated = calibrate_findings([raw], state_for(raw))[0]
    assert calibrated.evidence_status == 'source_grounded'
    assert calibrated.evidence_excerpt == '2: consume(codes);'
    assert 'not been reproduced' in calibrated.calibration_note
    assert raw.evidence_excerpt == 'invented code'


def test_wrong_file_and_unchanged_line_are_not_scored():
    bad_file = finding()
    bad_file.file_path = 'not-in-pr.ts'
    unchanged = finding()
    unchanged.line_start = unchanged.line_end = 1
    result = aggregator_node(state_for(bad_file, unchanged))
    assert result['verdict'].overall_score == 10
    assert all(f.evidence_status == 'outside_diff' for f in result['final_findings'])
    assert result['verdict'].score_breakdown['unverified_claims_excluded'] == 2


def test_compile_claim_needs_diagnostic_not_pattern_guess():
    raw = finding()
    raw.comment = 'useState([]) infers never[] so setCodes(body.codes) causes a compilation failure.'
    result = aggregator_node(state_for(raw))
    assert result['final_findings'][0].evidence_status == 'needs_verification'
    assert result['verdict'].overall_score == 10
    assert result['verdict'].score_breakdown['unverified_claims_excluded'] == 1


def test_distinct_bug_and_test_suggestion_on_same_line_are_not_merged():
    bug = finding()
    test = finding()
    test.kind = 'test_suggestion'
    test.category = 'test_coverage'
    assert len(_deduplicate([bug, test])) == 2


def test_review_coverage_discloses_prompt_bounds_and_no_execution():
    state = state_for()
    state.diff = 'x' * 10000
    state.review_mode = 'fast'
    state.review_stages = [{'reviewer': 'correctness', 'status': 'completed'}]
    result = review_coverage(state)
    stage = result['reviewers'][0]
    assert stage['diff_chars_included'] == 9000
    assert any('truncated' in note for note in stage['limitations'])
    assert any('not executed' in note for note in stage['limitations'])
    assert result['changed_file_paths'] == ['auth.ts']


def test_executor_withholds_unverified_comments(monkeypatch):
    raw = finding()
    state = state_for(raw)
    state.final_findings = [raw]
    state.human_approved = True
    state.verdict = Verdict(summary='Unverified concern', approved=True, overall_score=10, recommendation='comment')
    monkeypatch.setenv('GITHUB_TOKEN', 'fake')
    monkeypatch.setattr('nodes.executor._fetch_pr_head_sha', Mock(return_value='sha'))
    inline = Mock()
    issue = Mock(return_value=True)
    monkeypatch.setattr('nodes.executor._post_review_comment', inline)
    monkeypatch.setattr('nodes.executor._post_issue_comment', issue)
    receipt = executor_node(state)['posting_result']
    assert receipt['withheld_unverified'] == 1
    assert receipt['status'] == 'partially_posted'
    assert inline.call_count == 0
    assert issue.call_count == 1  # risk summary only


def test_posted_finding_does_not_claim_semantic_verification():
    raw = finding(evidence_status='source_grounded')
    body = _finding_body(raw)
    assert 'behavior not independently reproduced' in body
    assert 'uncalibrated' in body
