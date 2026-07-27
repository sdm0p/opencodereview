from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from typing import Optional

import requests

# --- ChromaDB needs to be imported with a guard because the ONNX runtime may
#     pull in extra dependencies on first load.
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None  # type: ignore[assignment]
    Collection = None  # type: ignore[assignment,misc]
    embedding_functions = None  # type: ignore[assignment]

from github_client import GitHubSession
from state import ContextChunk, OpenCodeReviewState

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
RAW_CONTENT_BASE = "https://raw.githubusercontent.com"

# How many chunks to retrieve per query
TOP_K = 15
# Max size (characters) for a single chunk
CHUNK_MAX_CHARS = 1_200
# How many merged PR discussions to fetch
MERGED_PR_LIMIT = 20

# Extensions we consider "source code" worth embedding
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".md", ".rst", ".txt",
    ".yml", ".yaml", ".toml", ".cfg", ".ini",
    ".sql", ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss",
}
# Files we always include if they exist
ALWAYS_INCLUDE = {
    "CONTRIBUTING.md", "CONTRIBUTING.rst",
    "README.md", "README.rst",
    "CODE_OF_CONDUCT.md",
    "STYLE_GUIDE.md",
    ".github/CONTRIBUTING.md",
}


# ─── Helpers: GitHub API ────────────────────────────────────────────────────


_GITHUB_SESSION: GitHubSession | None = None


def _gh_session() -> GitHubSession:
    """Return a lazily-initialized :class:`GitHubSession` for the retrieval
    node.  The session is cached across calls within the same PR review."""
    global _GITHUB_SESSION
    if _GITHUB_SESSION is None:
        _GITHUB_SESSION = GitHubSession()
    return _GITHUB_SESSION


def _get_json(url: str) -> Optional[dict | list]:
    """GET and return parsed JSON, or None on non-200, logging warnings.

    Uses :class:`GitHubSession` so 401 responses raise
    :class:`TokenRevokedError` automatically.
    """
    try:
        resp = _gh_session().get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("GitHub API %s returned %s", url.split("?")[0], resp.status_code)
        return None
    except requests.RequestException as exc:
        logger.warning("GitHub API request failed for %s: %s", url.split("?")[0], exc)
        return None


# ─── Helpers: chunking ──────────────────────────────────────────────────────


def _chunk_python_source(file_path: str, content: str) -> list[dict]:
    """Split Python source at top-level ``def`` / ``class`` / ``async def``."""
    chunks: list[dict] = []
    lines = content.splitlines(keepends=True)
    # Regex matches start of a top-level (non-indented) definition
    pattern = re.compile(r"^(async\s+)?(def |class )")
    current_start = 0
    for i, line in enumerate(lines):
        if i > 0 and pattern.match(line) and not line.startswith(" "):
            block = "".join(lines[current_start:i]).strip()
            if block:
                chunks.append({
                    "text": block,
                    "type": _classify_chunk(block),
                })
            current_start = i
    # Last block
    remaining = "".join(lines[current_start:]).strip()
    if remaining:
        chunks.append({"text": remaining, "type": _classify_chunk(remaining)})
    return chunks


def _chunk_generic(content: str) -> list[dict]:
    """Split text by blank lines into paragraphs, merging very short ones."""
    paragraphs = re.split(r"\n\s*\n", content)
    chunks: list[dict] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) < CHUNK_MAX_CHARS:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append({"text": buffer, "type": "paragraph"})
            buffer = para
    if buffer:
        chunks.append({"text": buffer, "type": "paragraph"})
    return chunks


def _classify_chunk(text: str) -> str:
    """Return ``function``, ``class``, or ``code``."""
    if text.startswith("class "):
        return "class"
    if text.startswith(("def ", "async def ")):
        return "function"
    return "code"


def _chunk_source(file_path: str, content: str) -> list[dict]:
    """Chunk a source file into semantically meaningful pieces."""
    if file_path.endswith(".py"):
        return _chunk_python_source(file_path, content)
    return _chunk_generic(content)


def _should_include(path: str) -> bool:
    """Return True for file paths we want to embed (source code, docs)."""
    if path in ALWAYS_INCLUDE:
        return True
    # Skip hidden files, test data, vendored deps
    parts = path.replace("\\", "/").split("/")
    if any(p.startswith(".") for p in parts):
        return False
    if "test" in path.lower() or "spec" in path.lower() or "__pycache__" in path:
        return False
    if any(p in ("venv", ".venv", "node_modules", "vendor", "dist", "build") for p in parts):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in SOURCE_EXTENSIONS


# ─── Helpers: building the vector store ─────────────────────────────────────


_VECTOR_DIR: str | None = None


def _ensure_vector_dir() -> str:
    """Return a persistent vector-store directory (not temp, so it survives
    across runs).  Override via OPENCODEREVIEW_VECTOR_DIR env var."""
    global _VECTOR_DIR
    if _VECTOR_DIR is None:
        _VECTOR_DIR = os.environ.get(
            "OPENCODEREVIEW_VECTOR_DIR",
            os.path.join(os.getcwd(), ".opencodereview", "vectors"),
        )
        os.makedirs(_VECTOR_DIR, exist_ok=True)
    return _VECTOR_DIR


def _repo_collection_name(repo: str) -> str:
    return f"repo_{repo.replace('/', '_')}"


def _get_or_build_collection(
    owner: str, repo_name: str,
    head_sha: str,
    exclude_paths: set[str],
) -> Collection:
    """Return a cached ChromaDB collection for ``repo``, rebuilding only when
    the base-branch SHA changes.

    The collection metadata key ``base_sha`` stores the SHA at build time.
    On subsequent calls, if the collection exists with a matching ``base_sha``
    it is returned immediately — no fetch, no chunk, no embedding.
    """
    if chromadb is None:
        raise RuntimeError("chromadb is not installed — cannot build vector store")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=_ensure_vector_dir())
    coll_name = _repo_collection_name(f"{owner}/{repo_name}")

    # ── Cache hit: collection exists with matching SHA ────────────────────
    try:
        existing = client.get_collection(name=coll_name, embedding_function=ef)
        meta = existing.metadata or {}
        if meta.get("base_sha") == head_sha:
            logger.info(
                "Vector store up-to-date (SHA=%s) — skipping rebuild",
                head_sha[:7],
            )
            return existing
        logger.info(
            "SHA mismatch: cached=%s, current=%s — rebuilding",
            (meta.get("base_sha") or "?"), head_sha[:7],
        )
        client.delete_collection(coll_name)
    except Exception:
        pass  # First build or error → create fresh

    return _build_vector_store(
        owner, repo_name, head_sha, exclude_paths, client, coll_name, ef,
    )


def _build_vector_store(
    owner: str, repo_name: str,
    head_sha: str,
    exclude_paths: set[str],
    client: chromadb.PersistentClient,
    coll_name: str,
    ef,
) -> Collection:
    """Fetch, chunk, embed, and index the non-changed files + past PRs.

    Each data source (codebase, past PRs, docs) is added to ChromaDB
    incrementally so a transient failure in one source does not lose
    progress from earlier sources.

    .. note::

        Prefer :func:`_get_or_build_collection` — this function always
        rebuilds from scratch.
    """
    collection = client.create_collection(
        name=coll_name,
        metadata={
            "hnsw:space": "cosine",
            "base_sha": head_sha,       # ← freshness key for cache lookups
            "repo": f"{owner}/{repo_name}",
        },
        embedding_function=ef,
    )

    total_added = 0

    # ---- 1. Source files from the repo (excluding changed files) ------------
    logger.info("Fetching repo file tree (base branch) …")
    tree_url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/git/trees/"
        f"{head_sha}?recursive=1"
    )
    tree_data = _get_json(tree_url)
    if tree_data and isinstance(tree_data, dict):
        tree_items: list[dict] = tree_data.get("tree", [])
        source_paths = [
            item["path"]
            for item in tree_items
            if item["type"] == "blob"
            and _should_include(item["path"])
            and item["path"] not in exclude_paths
        ]
        logger.info("Source files to embed (excl. changed): %d", len(source_paths))

        code_docs: list[str] = []
        code_metadatas: list[dict] = []
        code_ids: list[str] = []

        for fp in source_paths:
            raw_url = f"{RAW_CONTENT_BASE}/{owner}/{repo_name}/{head_sha}/{fp}"
            try:
                resp = requests.get(raw_url, timeout=30)
                if resp.status_code != 200:
                    continue
                content = resp.text
            except requests.RequestException:
                continue

            file_chunks = _chunk_source(fp, content)
            for chunk in file_chunks:
                code_docs.append(chunk["text"])
                code_metadatas.append({
                    "source": "codebase",
                    "file_path": fp,
                    "chunk_type": chunk["type"],
                })
                code_ids.append(str(uuid.uuid4()))

        if code_docs:
            collection.add(
                documents=code_docs,
                metadatas=code_metadatas,
                ids=code_ids,
            )
            total_added += len(code_docs)
            logger.info("Code chunks indexed: %d", len(code_docs))
    else:
        logger.warning("Could not fetch repo file tree — continuing with past PRs only")

    # ---- 2. Past merged PR discussions --------------------------------------
    prs_url = (
        f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/pulls"
        f"?state=closed&per_page={MERGED_PR_LIMIT}&sort=updated&direction=desc"
    )
    try:
        prs_data = _get_json(prs_url)
        if prs_data and isinstance(prs_data, list):
            pr_docs: list[str] = []
            pr_metadatas: list[dict] = []
            pr_ids: list[str] = []

            for pr_item in prs_data:
                if not pr_item.get("merged_at"):
                    continue
                pr_number = pr_item["number"]
                title = pr_item.get("title", "")
                body = pr_item.get("body") or ""
                text = f"PR #{pr_number}: {title}\n\n{body}".strip()
                if len(text) < 50:
                    continue
                chunks = _chunk_generic(text)
                for chunk in chunks:
                    pr_docs.append(chunk["text"])
                    pr_metadatas.append({
                        "source": "past_pr",
                        "file_path": f"PR #{pr_number}",
                        "chunk_type": "discussion",
                    })
                    pr_ids.append(str(uuid.uuid4()))

            if pr_docs:
                collection.add(
                    documents=pr_docs,
                    metadatas=pr_metadatas,
                    ids=pr_ids,
                )
                total_added += len(pr_docs)
                logger.info("Past-PR chunks indexed: %d", len(pr_docs))
    except Exception as exc:
        logger.warning("Failed to fetch past PRs (continuing without them): %s", exc)

    # ---- 3. CONTRIBUTING.md / docs ------------------------------------------
    doc_batch_docs: list[str] = []
    doc_batch_metas: list[dict] = []
    doc_batch_ids: list[str] = []

    for doc_path in ALWAYS_INCLUDE:
        raw_url = f"{RAW_CONTENT_BASE}/{owner}/{repo_name}/{head_sha}/{doc_path}"
        try:
            resp = requests.get(raw_url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 50:
                chunks = _chunk_generic(resp.text)
                for chunk in chunks:
                    doc_batch_docs.append(chunk["text"])
                    doc_batch_metas.append({
                        "source": "codebase",
                        "file_path": doc_path,
                        "chunk_type": "docs",
                    })
                    doc_batch_ids.append(str(uuid.uuid4()))
                logger.info("  Added docs: %s", doc_path)
        except requests.RequestException:
            pass

    if doc_batch_docs:
        collection.add(
            documents=doc_batch_docs,
            metadatas=doc_batch_metas,
            ids=doc_batch_ids,
        )
        total_added += len(doc_batch_docs)

    if total_added:
        logger.info(
            "Vector store built — %d total chunks in '%s'",
            total_added, coll_name,
        )
    else:
        logger.warning("Vector store is empty — no documents to index")

    return collection


# ─── Retrieval node ──────────────────────────────────────────────────────────


def retrieval_node(state: OpenCodeReviewState) -> dict:
    """Build (once per repo) a vector store of the codebase + past PRs, then
    query it with the current PR diff to find relevant context chunks.

    Returns state updates for ``context_chunks``.
    """
    # -- Bail out early if there is no diff to query against ------------------
    if not state.diff:
        logger.info("No diff available — skipping retrieval")
        return {}

    # -- Parse identifiers ----------------------------------------------------
    parts = state.repo.split("/")
    if len(parts) != 2:
        logger.warning("Invalid repo format '%s' — skipping retrieval", state.repo)
        return {}
    owner, repo_name = parts

    changed_paths: set[str] = {f.path for f in state.changed_files}

    # -- Build (or load) the vector store for this repo -----------------------
    # Use the base branch SHA as the cache key.  When re-reviewing the same
    # repo against the same base SHA the vector store is returned immediately
    # without re-fetching / re-embedding any files.
    base_sha = state.base_sha or "HEAD"  # fallback for synthetic / test state

    if chromadb is None:
        logger.warning("chromadb not available — returning empty context")
        return {}

    try:
        collection = _get_or_build_collection(
            owner, repo_name, base_sha, changed_paths,
        )
    except Exception as exc:
        logger.warning("Failed to build/get vector store: %s", exc)
        return {}

    # -- Query the store with the diff ----------------------------------------
    # Use a truncated version of the diff as the query for relevance
    query = state.diff
    if len(query) > 8_000:
        # Chroma has a max input length for the embedding model
        query = query[:8_000]

    try:
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K,
        )
    except Exception as exc:
        logger.warning("Vector store query failed: %s", exc)
        return {}

    # -- Build ContextChunk objects from results ------------------------------
    context_chunks: list[ContextChunk] = []
    if results and results.get("documents") and results["documents"][0]:
        docs_list = results["documents"][0]
        meta_list = results["metadatas"][0] if results.get("metadatas") else []
        dist_list = results["distances"][0] if results.get("distances") else []

        for i, doc_text in enumerate(docs_list):
            meta: dict = meta_list[i] if i < len(meta_list) else {}
            dist: float = dist_list[i] if i < len(dist_list) else 0.0
            # Convert distance to a relevance score (cosine distance → 1 - d)
            relevance = max(0.0, min(1.0, 1.0 - dist))

            context_chunks.append(ContextChunk(
                source=meta.get("source", "codebase"),
                file_path=meta.get("file_path", ""),
                content=doc_text,
                relevance_score=round(relevance, 3),
            ))

    logger.info(
        "Retrieved %d context chunks (query=%s)",
        len(context_chunks), _human_size(len(state.diff)),
    )

    return {
        "context_chunks": context_chunks,  # Annotated[..., add] → accumulated
    }


# ─── Utility ─────────────────────────────────────────────────────────────────


def _human_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KiB"
    else:
        return f"{bytes_ / (1024 * 1024):.1f} MiB"
