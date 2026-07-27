from __future__ import annotations

import enum
from operator import add
from typing import Annotated, Optional

from pydantic import BaseModel, Field


# ─── Sub-models ──────────────────────────────────────────────────────────────


class ChangedFile(BaseModel):
    """A single file modified in the PR."""
    path: str
    status: str  # "added" | "modified" | "removed" | "renamed"
    content: str  # Full file content after the PR
    diff_hunk: str  # The unified-diff hunk for this file


class ContextChunk(BaseModel):
    """A chunk of relevant context retrieved from the repo's codebase or past PRs."""
    source: str  # "codebase" | "past_pr"
    file_path: str  # Where the chunk came from
    content: str  # The relevant code/text
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Finding(BaseModel):
    """A single issue found during review."""
    file_path: str
    line_start: int
    line_end: int
    severity: Severity
    category: str  # "logic" | "security" | "style" | "performance" | "correctness"
    comment: str  # Human-readable explanation
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_fix: Optional[str] = None


class Verdict(BaseModel):
    """Aggregated final verdict across all findings."""
    summary: str
    approved: bool
    overall_score: float = Field(ge=0.0, le=10.0)
    critical_count: int = 0
    high_count: int = 0
    recommendation: str  # "approve" | "request_changes" | "block"


# ─── Graph State ─────────────────────────────────────────────────────────────


class OpenCodeReviewState(BaseModel):
    """Shared state for the OpenCodeReview LangGraph agent."""

    # PR metadata: set-once, single values
    repo: str = ""
    pr_number: int = 0
    base_sha: str = ""  # SHA of the base branch (used as vector-store cache key)
    diff: Optional[str] = None
    changed_files: list[ChangedFile] = Field(default_factory=list)

    # Retrieved context: accumulated across parallel retrieval nodes
    context_chunks: Annotated[list[ContextChunk], add] = Field(default_factory=list)

    # Review findings: accumulated across parallel review agents
    findings: Annotated[list[Finding], add] = Field(default_factory=list)

    # Aggregated (deduped, noise-filtered) findings: set-once by aggregator
    final_findings: list[Finding] = Field(default_factory=list)

    # Final verdict: set-once by the aggregation node
    verdict: Optional[Verdict] = None

    # Human-in-the-loop: set by the human_approval node
    human_approved: bool = False
