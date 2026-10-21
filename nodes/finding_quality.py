"""Deterministic source checks and conservative severity calibration.

Source grounding validates a location in the changed code; it is deliberately
not presented as execution, compilation, or semantic verification.
"""
from __future__ import annotations

import re

from state import Finding, OpenCodeReviewState, Severity


def patch_lines(patch: str) -> tuple[dict[int, str], set[int]]:
    source: dict[int, str] = {}
    added: set[int] = set()
    line = None
    for raw in patch.splitlines():
        if raw.startswith('@@'):
            match = re.search(r'\+(\d+)(?:,\d+)?', raw)
            line = int(match.group(1)) if match else None
        elif line is not None and raw.startswith(('+', ' ')):
            source[line] = raw[1:]
            if raw.startswith('+'):
                added.add(line)
            line += 1
    return source, added


def calibrate_findings(findings: list[Finding], state: OpenCodeReviewState) -> list[Finding]:
    files = {item.path: item for item in state.changed_files}
    results = []
    for original in findings:
        finding = original.model_copy(deep=True)
        notes = []
        category = finding.category.lower()
        if category in {'test_coverage', 'test coverage', 'testing', 'tests'} or finding.kind == 'test_suggestion':
            finding.kind = 'test_suggestion'
            finding.severity = Severity.LOW if finding.severity != Severity.INFO else Severity.INFO
            notes.append('Test suggestion, not an established runtime defect; excluded from the risk score. Test availability outside the supplied context is unknown.')
        else:
            finding.kind = 'security_concern' if category == 'security' else 'bug'

        file = files.get(finding.file_path)
        finding.evidence_excerpt = ''  # Never trust an unverified model-supplied quote.
        finding.evidence_status = 'outside_diff'
        if file and finding.line_start > 0 and finding.line_end >= finding.line_start:
            source, added = patch_lines(file.diff_hunk)
            content_lines = file.content.splitlines()
            start, end = finding.line_start, finding.line_end
            location_exists = end <= len(content_lines) if content_lines else all(i in source for i in range(start, end + 1))
            overlaps_change = any(start <= number <= end for number in added)
            if location_exists and overlaps_change:
                finding.evidence_status = 'source_grounded'
                excerpt = []
                for number in range(start, min(end + 1, start + 12)):
                    text = content_lines[number - 1] if content_lines else source[number]
                    excerpt.append(f'{number}: {text}')
                finding.evidence_excerpt = '\n'.join(excerpt)[:1800]
                notes.append('Location matches changed source. The claimed behavior has not been reproduced or independently verified.')
            else:
                notes.append('Location could not be matched to added/modified lines in the supplied diff; excluded from the risk score.')
        else:
            notes.append('File or line range is outside the supplied changed code; excluded from the risk score.')

        # Source text alone cannot prove compiler outcomes. In particular,
        # response.json() is commonly any, so useState([]) is not itself proof
        # of the reported TypeScript assignment error.
        if finding.kind != 'test_suggestion' and re.search(
            r'compil(?:ation|e|ing)|type[- ]?check|type[- ]?error|will not build|cannot build|never\[\]',
            finding.comment, re.IGNORECASE,
        ):
            finding.evidence_status = 'needs_verification'
            notes.append('Compiler/type-system claim requires a reproducible compiler diagnostic; none was run by this review. Excluded from the risk score.')
        finding.calibration_note = ' '.join(notes)
        results.append(finding)
    return results


def review_coverage(state: OpenCodeReviewState) -> dict:
    """Report actual input bounds without claiming all source was inspected."""
    limits = {'correctness': (9000, 600, 2000), 'security': (None, 4000, 6000), 'test_coverage': (None, 4000, 6000)}
    reviewers = []
    for stage in state.review_stages:
        reviewer = stage.get('reviewer', '')
        diff_limit, file_limit, context_limit = limits.get(reviewer, (None, None, None))
        limitations = ['Source checks validate location only; code and tests were not executed.']
        if diff_limit and len(state.diff or '') > diff_limit:
            limitations.append('Diff truncated in this reviewer prompt.')
        if file_limit and any(len(file.content) > file_limit for file in state.changed_files):
            limitations.append('Changed-file context has a per-file character limit.')
        if state.review_mode == 'fast':
            limitations.append('Fast mode supplies windows around changed lines rather than complete files.')
        if sum(len(chunk.content) for chunk in state.context_chunks) > (context_limit or 0):
            limitations.append('Retrieved context truncated to the reviewer context limit.')
        reviewers.append({**stage, 'diff_chars_included': min(len(state.diff or ''), diff_limit) if diff_limit else len(state.diff or ''),
                          'file_context_limit_chars': file_limit, 'context_limit_chars': context_limit, 'limitations': limitations})
    return {'mode': state.review_mode, 'incremental': bool(state.incremental_base_sha),
            'files_changed': len(state.changed_files), 'files_available': sum(bool(file.content) for file in state.changed_files),
            'changed_file_paths': [file.path for file in state.changed_files], 'diff_chars': len(state.diff or ''),
            'reviewers': reviewers, 'limitations': ['Review quality is not proven by the risk score. Missing repository context can hide issues.']}
