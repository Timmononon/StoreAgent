"""Evidence coverage selection uses source identities, never benchmark labels."""

from copy import deepcopy
import asyncio

import pytest

from backend.rag.ranking import fuse_ranked_documents, select_diverse_documents
from backend.rag.profiles import RagRetrievalProfile
from test_rag_latency_guards import load_utils


def _doc(source: str, chunk: int = 1, **extra) -> dict:
    return {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-a",
        "document_id": source,
        "document_version_id": f"{source}-v1",
        "chunk_id": f"{source}-{chunk}",
        "filename": f"{source}.html",
        "text": f"evidence {source} {chunk}",
        **extra,
    }


def test_rrf_combines_branch_votes_without_counting_duplicates_in_one_branch():
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    groups = [[a, a, b], [c, b]]
    before = deepcopy(groups)

    result = fuse_ranked_documents(groups)

    assert [doc["chunk_id"] for doc in result] == ["b-1", "a-1", "c-1"]
    assert result[0]["score"] == pytest.approx(1 / 62 + 1 / 62)
    assert [doc["rrf_rank"] for doc in result] == [1, 2, 3]
    assert groups == before
    assert result == fuse_ranked_documents(list(reversed(groups)))


def test_fusion_preserves_document_version_identity_and_discards_query_local_scores():
    first = _doc("a", rerank_score=0.99)
    other = _doc("b", chunk_id=first["chunk_id"], rerank_score=0.01)
    revised = _doc("a", document_version_id="a-v2", rerank_score=0.5)

    result = fuse_ranked_documents([[first], [other], [revised]])

    assert len(result) == 3
    assert all("rerank_score" not in doc for doc in result)
    assert {doc["document_version_id"] for doc in result} == {"a-v1", "b-v1", "a-v2"}


def test_source_priority_reaches_later_documents_then_fills_with_relevant_chunks():
    docs = [_doc("a", i) for i in range(1, 8)] + [_doc("b"), _doc("c")]
    before = deepcopy(docs)

    selected = select_diverse_documents(docs, top_k=5)

    assert [doc["chunk_id"] for doc in selected] == ["a-1", "b-1", "c-1", "a-2", "a-3"]
    assert docs == before
    assert select_diverse_documents(docs[:4], top_k=10) == docs[:4]
    assert select_diverse_documents([], top_k=10) == []


def test_source_priority_uses_full_scope_and_does_not_group_unknown_sources_together():
    docs = [
        _doc("a"),
        _doc("a", 2),
        _doc("a", 3, tenant_id="tenant-b"),
        _doc("a", 4, document_version_id="a-v2"),
        {"chunk_id": "unknown-1", "text": "one"},
        {"chunk_id": "unknown-2", "text": "two"},
    ]
    assert [doc["chunk_id"] for doc in select_diverse_documents(docs, top_k=5)] == [
        "a-1",
        "a-3",
        "a-4",
        "unknown-1",
        "unknown-2",
    ]


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_selection_rejects_invalid_limits(top_k):
    with pytest.raises(ValueError):
        select_diverse_documents([_doc("a")], top_k=top_k)


def _utils():
    utils, _embedding = load_utils({"AUTO_MERGE_ENABLED": "false"})
    return utils


def test_combination_reranks_the_bounded_union_with_the_original_question():
    utils = _utils()
    calls = []

    def rerank(**kwargs):
        calls.append(kwargs)
        docs = kwargs["docs"]
        return list(reversed(docs)), {
            "rerank_applied": True,
            "post_rerank_count": len(docs),
        }

    utils._rerank_documents = rerank
    groups = [[_doc("a", i) for i in range(1, 5)], [_doc("b"), _doc("c")]]
    profile = RagRetrievalProfile(top_k=3, candidate_k=5)

    def cancellation():
        return False

    result = utils.combine_retrieval_results(
        "original question",
        groups,
        profile=profile,
        deadline=123.0,
        cancellation=cancellation,
    )

    assert len(calls) == 1
    assert calls[0]["query"] == "original question"
    assert len(calls[0]["docs"]) == 5
    assert calls[0]["top_k"] == 5  # Keep every bounded score until source selection.
    assert calls[0]["deadline"] == 123.0
    assert calls[0]["cancellation"] is cancellation
    assert len(result["docs"]) == 3
    assert len({doc["document_id"] for doc in result["docs"]}) == 3
    assert result["meta"]["deduplicated_recall_count"] == 6
    assert result["meta"]["retrieval_top_k"] == 3


def test_dense_and_hybrid_use_the_same_fusion_selection_without_calling_rerank():
    utils = _utils()

    def unexpected_rerank(**kwargs):
        pytest.fail("a profile-disabled reranker must not be called")

    utils._rerank_documents = unexpected_rerank
    groups = [[_doc("a", i) for i in range(1, 5)], [_doc("b"), _doc("c")]]
    results = [
        utils.combine_retrieval_results(
            "original",
            groups,
            profile=RagRetrievalProfile(
                mode=mode, rerank_enabled=False, top_k=3, candidate_k=6
            ),
        )
        for mode in ("dense", "hybrid")
    ]
    assert results[0]["docs"] == results[1]["docs"]
    assert {doc["document_id"] for doc in results[0]["docs"]} == {"a", "b", "c"}
    assert all(
        result["meta"]["rerank_skip_reason"] == "profile_disabled" for result in results
    )


def test_combination_keeps_rerank_fallback_explicit_and_does_not_change_input():
    utils = _utils()
    groups = [[_doc("a", i) for i in range(1, 4)], [_doc("b")]]
    before = deepcopy(groups)
    utils._rerank_documents = lambda **kwargs: (
        kwargs["docs"],
        {
            "rerank_applied": False,
            "rerank_fallback_applied": True,
            "rerank_error_code": "RERANK_TIMEOUT",
            "rerank_attempts": 2,
            "rerank_threshold_applied": False,
        },
    )

    result = utils.combine_retrieval_results("question", groups, top_k=3)

    assert result["meta"]["rerank_fallback_applied"] is True
    assert result["meta"]["rerank_error_code"] == "RERANK_TIMEOUT"
    assert result["meta"]["rerank_threshold_applied"] is False
    assert [doc["chunk_id"] for doc in result["docs"]] == ["a-1", "b-1", "a-2"]
    assert groups == before


def test_cancelled_selection_propagates_even_when_rerank_is_disabled():
    utils = _utils()
    with pytest.raises(asyncio.CancelledError):
        utils.combine_retrieval_results(
            "question",
            [[_doc("a")]],
            profile=RagRetrievalProfile(rerank_enabled=False),
            cancellation=lambda: True,
        )


def test_empty_union_does_not_call_rerank_or_create_evidence():
    utils = _utils()
    utils._rerank_documents = lambda **kwargs: pytest.fail("no empty provider calls")
    result = utils.combine_retrieval_results("question", [[], []])
    assert result["docs"] == []
    assert result["meta"]["rerank_skip_reason"] == "no_candidates"
    assert result["meta"]["retrieval_empty"] is True
