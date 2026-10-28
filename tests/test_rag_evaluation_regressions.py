import json
from unittest.mock import patch

import pytest

import backend.evaluation.runtime as runtime_module
from backend.evaluation.rag import RagEvalCase, RagExpectedBehavior, RagOutcome
from backend.evaluation.rag_adapters import observation_from_rag_results
from backend.evaluation.runtime import RagEvaluationRuntime
from backend.providers import ProviderError, ProviderExecutor
from test_rag_evaluation_judge_runtime import _EvaluatorModel, _Models
from test_rag_short_circuit import FakeStructuredModel, _doc, _meta, load_pipeline
from tests.support import TEST_MODEL_SNAPSHOT


def _grade(**changes):
    return {
        "relevance": "strong",
        "answerability": "sufficient",
        "ambiguity": "none",
        "route": "answer",
        "confidence": 0.95,
        **changes,
    }


@pytest.mark.parametrize(
    "answerability,expected",
    [("sufficient", "ANSWERABLE"), ("partial", "INSUFFICIENT_EVIDENCE")],
)
def test_merged_partial_branches_are_assessed_against_the_whole_question(
    answerability, expected
):
    prompts = []
    pipeline = load_pipeline(retrieve_documents=lambda query: {})

    def grade(schema, prompt):
        prompts.append(prompt)
        return _grade(answerability=answerability)

    pipeline._get_grader_model = lambda *_: FakeStructuredModel(grade)
    state = pipeline._initial_state(
        "东区采用哪个套餐，该套餐响应时间是多少？", tenant_id="tenant-a"
    )
    state["sub_results"] = [
        {
            "question": "东区套餐",
            "docs": [_doc("东区采用 Plus 套餐。", "site")],
            "retrieval_status": "partial",
        },
        {
            "question": "套餐响应时间",
            "docs": [_doc("Plus 套餐响应时间为两小时。", "plan")],
            "retrieval_status": "partial",
        },
    ]
    result = pipeline.synthesis(state)
    assert len(prompts) == 1
    assert state["question"] in prompts[0]
    assert "东区采用 Plus" in prompts[0] and "两小时" in prompts[0]
    assert result["retrieval_outcome"] == expected
    assert len(result["docs"]) == 2
    if expected == "ANSWERABLE":
        assert result["rag_trace"]["coverage_gap_questions"] == []


def test_healthy_search_with_explicitly_no_usable_evidence_is_no_knowledge():
    pipeline = load_pipeline(
        retrieve_documents=lambda query, top_k=5: {
            "docs": [_doc("设备重 20 千克。")],
            "meta": _meta(1),
        }
    )
    pipeline._get_grader_model = lambda *_: FakeStructuredModel(
        lambda schema, prompt: _grade(
            relevance="weak", answerability="none", route="no_knowledge"
        )
    )
    result = pipeline.checkpointless_rag_graph.invoke(
        pipeline._initial_state("设备外壳的色号是多少？", tenant_id="tenant-a")
    )
    assert result["route"] == "no_knowledge"
    assert result["retrieval_outcome"] == "NO_KNOWLEDGE"
    assert result["docs"] == result["rag_trace"]["retrieved_chunks"] == []


@pytest.mark.parametrize(
    "relevance,route", [("none", "no_knowledge"), ("weak", "insufficient_evidence")]
)
def test_rejected_evidence_is_removed_from_both_answer_and_scoring(relevance, route):
    pipeline = load_pipeline(retrieve_documents=lambda query: {})
    pipeline._get_grader_model = lambda *_: FakeStructuredModel(
        lambda schema, prompt: _grade(relevance=relevance, answerability="none")
    )
    doc = _doc("未包含所需依据的候选。")
    state = pipeline._initial_state("证据问题？", tenant_id="tenant-a")
    state.update(
        docs=[doc],
        context=doc["text"],
        rag_trace={"retrieved_chunks": [doc], "initial_retrieved_chunks": [doc]},
    )
    state.update(pipeline.grade_documents_node(state))
    assert state["route"] == route
    assert state["docs"] == state["rag_trace"]["retrieved_chunks"] == []
    case = RagEvalCase(
        id="rejected", question=state["question"], expected=RagExpectedBehavior()
    )
    observation = observation_from_rag_results(
        case, initial=state, final=state, duration_ms=1
    )
    assert not observation.retrieved_chunks
    assert len(observation.initial_retrieved_chunks) == 1


def test_unspecified_referent_uses_clarification_even_with_candidate_options():
    pipeline = load_pipeline(
        retrieve_documents=lambda query, top_k=5: {
            "docs": [_doc("Atlas 和 Vega 的参数。")],
            "meta": _meta(1),
        }
    )
    pipeline._get_grader_model = lambda *_: FakeStructuredModel(
        lambda schema, prompt: _grade(
            answerability="partial",
            ambiguity="multiple_candidates",
            route="scope_select",
            hitl_options=["Atlas", "Vega"],
        )
    )
    result = pipeline.checkpointless_rag_graph.invoke(
        pipeline._initial_state("它的功率是多少？", tenant_id="tenant-a")
    )
    assert result["route"] == "clarify"
    assert result["retrieval_status"] == "needs_clarification"


def test_current_value_plus_change_history_does_not_take_single_fact_shortcut():
    prompts = []
    pipeline = load_pipeline(retrieve_documents=lambda query: {})

    def planner(schema, prompt):
        prompts.append(prompt)
        return {"complexity": "complex", "sub_questions": ["当前功率", "功率调整版本"]}

    pipeline._get_complexity_model = lambda *_: FakeStructuredModel(planner)
    result = pipeline.classify_complexity(
        pipeline._initial_state(
            "截至 Atlas R3，最新版本的功率是多少，哪次版本更新改变了功率？",
            tenant_id="tenant-a",
        )
    )
    assert len(prompts) == 1
    assert result["complexity"] == "complex"


def test_judge_receives_expected_abstention_without_automatically_passing_it():
    prompts = []

    class CapturingEvaluator(_EvaluatorModel):
        def invoke(self, messages):
            prompts.append(messages[0]["content"])
            return super().invoke(messages)

    models = _Models()
    models.evaluator = CapturingEvaluator()
    case = RagEvalCase(
        id="unknown-color",
        question="外壳色号？",
        expected=RagExpectedBehavior(
            route="no_knowledge", outcome="NO_KNOWLEDGE", acceptable_abstention=True
        ),
    )
    runtime = RagEvaluationRuntime(
        models=models, executor=ProviderExecutor(sleeper=lambda _: None)
    )
    result = {
        "route": "no_knowledge",
        "retrieval_outcome": "NO_KNOWLEDGE",
        "docs": [],
        "rag_trace": {"retrieved_chunks": []},
    }
    with patch.object(runtime_module, "run_rag_graph", return_value=result):
        execution = runtime.execute_case(
            job_id="eval-test",
            case=case,
            model_snapshot=TEST_MODEL_SNAPSHOT,
            timeout_seconds=30,
            cancellation=lambda: False,
        )
    payload = json.loads(prompts[0][prompts[0].index("{") :])
    assert payload["expected_outcome"] == "NO_KNOWLEDGE"
    assert payload["acceptable_abstention"] is True
    assert "拒答" in prompts[0]
    assert execution.observation.outcome is RagOutcome.NO_KNOWLEDGE
    assert execution.observation.judge.answer_correctness == 0.95


def test_multi_fact_planner_cannot_silently_return_a_simple_plan():
    pipeline = load_pipeline(retrieve_documents=lambda query: {})
    pipeline._get_complexity_model = lambda *_: FakeStructuredModel(
        lambda schema, prompt: {"complexity": "simple", "sub_questions": []}
    )
    with pytest.raises(ProviderError):
        pipeline.classify_complexity(
            pipeline._initial_state(
                "南区部署了多少台 Atlas，采用哪个维护计划，该计划多久响应？",
                tenant_id="tenant-a",
            )
        )


@pytest.mark.parametrize(
    "question",
    [
        "哪个 Atlas 维护计划没有设备上限，响应时间是多少？",
        "Atlas R3 中哪个选项被弃用，替代选项是什么？",
    ],
)
def test_one_record_lookup_may_remain_simple_with_two_question_words(question):
    prompts = []
    pipeline = load_pipeline(retrieve_documents=lambda query: {})

    def planner(schema, prompt):
        prompts.append(prompt)
        return {"complexity": "simple", "sub_questions": []}

    pipeline._get_complexity_model = lambda *_: FakeStructuredModel(planner)
    result = pipeline.classify_complexity(
        pipeline._initial_state(question, tenant_id="tenant-a")
    )
    assert len(prompts) == 1
    assert result["complexity"] == "simple"
