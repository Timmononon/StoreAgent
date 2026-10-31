"""Deterministic fusion and bounded source selection for production Evidence."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any


def _source_identity(document: Mapping[str, Any]) -> tuple[str, ...]:
    scope = tuple(
        str(document.get(field) or "")
        for field in (
            "tenant_id",
            "knowledge_base_id",
            "document_version_id",
            "index_version",
        )
    )
    source = str(document.get("document_id") or document.get("filename") or "")
    if not source:
        source = (
            str(document.get("chunk_id") or "")
            or hashlib.sha256(str(document.get("text") or "").encode()).hexdigest()
        )
    return (*scope, source)


def _chunk_identity(document: Mapping[str, Any]) -> tuple[str, ...]:
    chunk = str(document.get("chunk_id") or "")
    if not chunk:
        chunk = hashlib.sha256(
            f"{document.get('page_number')}|{document.get('text') or ''}".encode()
        ).hexdigest()
    return (*_source_identity(document), chunk)


def fuse_ranked_documents(
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Fuse query-local rankings using RRF (k=60), with stable identity ties.

    Query-local reranker scores are not comparable across subquestions. Replace
    their recall scores with the fusion score and let the original-question
    reranker assign fresh scores. A duplicate in one list contributes one vote.
    """
    representatives: dict[tuple[str, ...], dict[str, Any]] = {}
    contributions: dict[tuple[str, ...], list[float]] = defaultdict(list)
    best_ranks: dict[tuple[str, ...], int] = {}
    for group in groups:
        seen: set[tuple[str, ...]] = set()
        for document in group:
            key = _chunk_identity(document)
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            representatives.setdefault(key, dict(document))
            contributions[key].append(1.0 / (60 + rank))
            best_ranks[key] = min(best_ranks.get(key, rank), rank)

    scores = {key: math.fsum(values) for key, values in contributions.items()}
    keys = sorted(representatives, key=lambda key: (-scores[key], best_ranks[key], key))
    fused = []
    for rank, key in enumerate(keys, 1):
        document = representatives[key]
        document.pop("rerank_score", None)
        document.update(score=scores[key], rrf_rank=rank)
        fused.append(document)
    return fused


def select_diverse_documents(
    documents: Sequence[Mapping[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    """Prefer one chunk per source, then fill unused slots in relevance order.

    This is a soft priority, so a single-source answer can still use multiple
    chunks. Selection reads identities and current order, never gold labels.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive int")
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for document in documents:
        key = _source_identity(document)
        if key in seen:
            deferred.append(dict(document))
            continue
        seen.add(key)
        selected.append(dict(document))
        if len(selected) == top_k:
            return selected
    selected.extend(deferred[: top_k - len(selected)])
    return selected
