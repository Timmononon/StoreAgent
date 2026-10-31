from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.providers import (
    ProviderCallContext,
    ProviderCode,
    ProviderError,
    ProviderOperation,
)
from test_rag_short_circuit import _doc, load_pipeline


def _pipeline():
    return load_pipeline(retrieve_documents=lambda **kwargs: {"docs": [], "meta": {}})


def _branch(question, docs, status="answerable"):
    return {"question": question, "docs": docs, "retrieval_status": status}


def _docs(source, count):
    return [
        {
            **_doc(f"{source} evidence {i}", f"{source}-{i}", f"{source}.html"),
            "document_id": source,
            "document_version_id": f"{source}-v1",
        }
        for i in range(count)
    ]


def _grade(pipeline, *, sufficient):
    return (
        pipeline.EvidenceGrade(
            relevance="strong",
            answerability="sufficient" if sufficient else "partial",
            ambiguity="none",
            route="answer",
            confidence=0.8,
        ),
        SimpleNamespace(characters=100, omitted_count=0, truncated_count=0),
    )


@pytest.mark.parametrize("sufficient", [False, True])
def test_bounded_selection_regrades_actual_evidence_before_claiming_full_coverage(
    sufficient,
):
    pipeline = _pipeline()
    state = {
        "question": "original multi-source question",
        "sub_questions": ["a", "b"],
        "sub_results": [_branch("b", _docs("b", 4)), _branch("a", _docs("a", 4))],
    }
    pipeline._evaluate_evidence = Mock(
        return_value=_grade(pipeline, sufficient=sufficient)
    )

    result = pipeline.synthesis(state)

    assert len(result["docs"]) == 5
    assert result["docs"] == result["rag_trace"]["retrieved_chunks"]
    pipeline._evaluate_evidence.assert_called_once_with(state, result["docs"])
    assert {doc["document_id"] for doc in result["docs"]} == {"a", "b"}
    selected = {doc["chunk_id"] for doc in result["docs"]}
    for doc in _docs("a", 4) + _docs("b", 4):
        assert (doc["text"] in result["context"]) is (doc["chunk_id"] in selected)
    assert result["retrieval_outcome"] == (
        "ANSWERABLE" if sufficient else "INSUFFICIENT_EVIDENCE"
    )
    assert result["rag_trace"]["coverage_gap_questions"] == (
        [] if sufficient else [state["question"]]
    )


def test_branch_completion_order_cannot_change_selected_evidence_or_coverage_gaps():
    pipeline = _pipeline()
    pipeline._evaluate_evidence = Mock(return_value=_grade(pipeline, sufficient=False))
    branches = [_branch("a", _docs("a", 4)), _branch("b", _docs("b", 4))]
    state = {
        "question": "original",
        "sub_questions": ["a", "b"],
        "sub_results": branches,
    }
    first = pipeline.synthesis(state)
    second = pipeline.synthesis({**state, "sub_results": list(reversed(branches))})
    assert first == second


def test_healthy_empty_branch_is_not_upgraded_by_joint_grading():
    pipeline = _pipeline()
    pipeline._evaluate_evidence = Mock(
        side_effect=AssertionError("must keep missing branch")
    )
    result = pipeline.synthesis(
        {
            "question": "original",
            "sub_questions": ["a", "b"],
            "sub_results": [
                _branch("a", _docs("a", 2)),
                _branch("b", [], "no_knowledge"),
            ],
        }
    )
    assert result["retrieval_outcome"] == "INSUFFICIENT_EVIDENCE"
    assert result["rag_trace"]["coverage_gap_questions"] == ["b"]


def test_global_rerank_fallback_is_retained_in_trace_and_warning():
    pipeline = _pipeline()
    docs = _docs("a", 1)
    meta = {
        "deduplicated_recall_count": 1,
        "rerank_error_code": "RERANK_TIMEOUT",
        "rerank_fallback_applied": True,
        "rerank_attempts": 2,
    }
    pipeline.combine_retrieval_results = Mock(return_value={"docs": docs, "meta": meta})
    pipeline._emit_retrieval_warnings = Mock()
    result = pipeline.synthesis(
        {
            "question": "original",
            "sub_results": [_branch("a", docs)],
        }
    )
    assert result["rag_trace"]["rerank_fallback_applied"] is True
    assert result["rag_trace"]["rerank_error_code"] == "RERANK_TIMEOUT"
    pipeline._emit_retrieval_warnings.assert_called_once()


def test_all_failed_branches_propagate_before_any_global_rerank():
    pipeline = _pipeline()
    error = ProviderError(
        ProviderCode.EMBEDDING_UNAVAILABLE,
        context=ProviderCallContext(
            provider="embedding", operation=ProviderOperation.EMBEDDING
        ),
    )
    pipeline.combine_retrieval_results = Mock(
        side_effect=AssertionError("must propagate")
    )
    with pytest.raises(ProviderError) as failure:
        pipeline.synthesis(
            {
                "question": "original",
                "sub_results": [
                    {
                        "question": "a",
                        "retrieval_status": "provider_failed",
                        "docs": [],
                        "provider_error": error.to_snapshot(),
                    }
                ],
            }
        )
    assert failure.value.code == ProviderCode.EMBEDDING_UNAVAILABLE
