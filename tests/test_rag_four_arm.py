from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from pydantic import ValidationError

from backend.evaluation.experiments import experiment_definition
from backend.rag.profiles import RagRetrievalProfile
from backend.runs.request_context import RunRequestContext
from test_rag_latency_guards import load_utils


@pytest.mark.parametrize(
    "name,mode,rerank,strategy",
    [
        ("dense_baseline", "dense", False, "concatenate_v1"),
        ("hybrid_baseline", "hybrid", False, "concatenate_v1"),
        ("hybrid_rerank_baseline", "hybrid", True, "concatenate_v1"),
        ("hybrid_rerank_optimized", "hybrid", True, "source_priority_v1"),
        ("dense", "dense", False, "source_priority_v1"),
        ("hybrid", "hybrid", False, "source_priority_v1"),
        ("hybrid_rerank", "hybrid", True, "source_priority_v1"),
    ],
)
def test_frozen_names_unambiguously_select_strategy_without_reinterpreting_history(
    name, mode, rerank, strategy
):
    definition = experiment_definition(name)
    assert (
        definition.mode,
        definition.rerank_enabled,
        definition.synthesis_strategy,
    ) == (mode, rerank, strategy)


def _docs():
    return [
        [{"chunk_id": f"a-{i}", "document_id": "a", "text": f"a{i}"} for i in range(4)],
        [{"chunk_id": "b-1", "document_id": "b", "text": "b"}],
    ]


def test_c0_keeps_original_concatenation_without_any_second_rerank():
    utils, _ = load_utils({"AUTO_MERGE_ENABLED": "false"})
    utils._rerank_documents = lambda **kwargs: pytest.fail("C0 cannot globally rerank")
    result = utils.combine_retrieval_results(
        "original question",
        _docs(),
        profile=RagRetrievalProfile(top_k=3, candidate_k=6),
        synthesis_strategy="concatenate_v1",
    )
    assert [doc["chunk_id"] for doc in result["docs"]] == [
        "a-0",
        "a-1",
        "a-2",
        "a-3",
        "b-1",
    ]
    assert [doc["rrf_rank"] for doc in result["docs"]] == list(range(1, 6))
    assert result["meta"]["retrieval_pipeline"] == "subquery_concat_v1"


def test_explicit_strategies_are_isolated_across_overlapping_requests():
    utils, _ = load_utils({"AUTO_MERGE_ENABLED": "false"})
    barrier = threading.Barrier(2)
    profile = RagRetrievalProfile(top_k=3, candidate_k=6, rerank_enabled=False)

    def run(strategy):
        context = RunRequestContext.for_sync(
            user_id=strategy, thread_id=strategy, tenant_id="test"
        )
        try:
            context.configure_rag_retrieval_profile(
                profile, synthesis_strategy=strategy
            )
            barrier.wait(timeout=5)
            return utils.combine_retrieval_results(
                "original",
                _docs(),
                profile=context.rag_retrieval_profile,
                synthesis_strategy=context.rag_synthesis_strategy,
            )["docs"]
        finally:
            context.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline = pool.submit(run, "concatenate_v1")
        optimized = pool.submit(run, "source_priority_v1")
        assert len(baseline.result()) == 5
        selected = optimized.result()
        assert len(selected) == 3
        assert {doc["document_id"] for doc in selected} == {"a", "b"}


def test_context_refuses_strategy_rebinding_with_identical_retrieval_parameters():
    context = RunRequestContext.for_sync(user_id="a", thread_id="a")
    profile = RagRetrievalProfile()
    try:
        context.configure_rag_retrieval_profile(
            profile, synthesis_strategy="concatenate_v1"
        )
        with pytest.raises(ValueError, match="rebound"):
            context.configure_rag_retrieval_profile(
                profile, synthesis_strategy="source_priority_v1"
            )
    finally:
        context.close()


def test_frozen_snapshot_rejects_rerank_or_mode_inconsistent_with_arm():
    from backend.evaluation.experiments import RagExperimentSnapshot

    with pytest.raises(ValidationError):
        RagExperimentSnapshot(
            name="hybrid_rerank_baseline",
            tenant_id="test",
            knowledge_base_id="kb",
            corpus_fingerprint="a" * 64,
            index_id="b" * 64,
            document_count=1,
            runtime_fingerprint="c" * 64,
            retrieval=RagRetrievalProfile(rerank_enabled=False),
        )
