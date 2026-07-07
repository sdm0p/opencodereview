from __future__ import annotations

import logging
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    aggregator_node,
    correctness_reviewer_node,
    executor_node,
    human_approval_node,
    ingestion_node,
    retrieval_node,
    security_reviewer_node,
    test_coverage_reviewer_node,
)
from state import OpenCodeReviewState

logger = logging.getLogger(__name__)


def build_graph(db_path: str = "checkpoints.db") -> StateGraph:
    """Build and compile the OpenCodeReview graph with SqliteSaver checkpointing.

    Topology:

        START → ingestion → retrieval
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
           correctness    security    test_coverage
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                            aggregator
                                  │
                                  ▼
                         human_approval  ←── interrupt() pauses here
                                  │
                                  ▼
                             executor
                           ┌────┴────┐
                 ┌────────▼─▼────────┐
                 │  Inline review    │
                 │  comments at file │
                 │  /line positions  │
                 │  422 → issue cmt  │
                 └────────┬─┬────────┘
                           │
                          END
    """
    builder = StateGraph(OpenCodeReviewState)

    # -- Nodes ----------------------------------------------------------------
    builder.add_node("ingestion", ingestion_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("correctness_review", correctness_reviewer_node)
    builder.add_node("security_review", security_reviewer_node)
    builder.add_node("test_coverage_review", test_coverage_reviewer_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("executor", executor_node)

    # -- Linear prefix --------------------------------------------------------
    builder.add_edge(START, "ingestion")
    builder.add_edge("ingestion", "retrieval")

    # -- Fan-out: retrieval → all three reviewers in parallel -----------------
    builder.add_edge("retrieval", "correctness_review")
    builder.add_edge("retrieval", "security_review")
    builder.add_edge("retrieval", "test_coverage_review")

    # -- Fan-in: all three reviewers → aggregator -----------------------------
    builder.add_edge("correctness_review", "aggregator")
    builder.add_edge("security_review", "aggregator")
    builder.add_edge("test_coverage_review", "aggregator")

    # -- HITL + posting -------------------------------------------------------
    builder.add_edge("aggregator", "human_approval")
    builder.add_edge("human_approval", "executor")
    builder.add_edge("executor", END)

    # -- Checkpointer (required for interrupt/resume to work) -----------------
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("state", "ChangedFile"),
        ("state", "ContextChunk"),
        ("state", "Severity"),
        ("state", "Finding"),
        ("state", "Verdict"),
    ])

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn, serde=serde)

    return builder.compile(checkpointer=saver)
