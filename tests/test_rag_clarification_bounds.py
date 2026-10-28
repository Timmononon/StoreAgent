import unittest

from backend.rag.checkpoint_runner import _compact_hitl_state
from backend.runs.request_context import RunRequestContext
from backend.tools.knowledge import _render_rag_result
from test_rag_short_circuit import FakeStructuredModel, _doc, _meta, load_pipeline


class RagClarificationBoundsTests(unittest.TestCase):
    def pipeline(self, grades, *, complex_question=False):
        calls = {"retrieve": [], "prompts": []}

        def retrieve(query, top_k=5):
            calls["retrieve"].append(query)
            return {"docs": [_doc("HMM was fitted four times.")], "meta": _meta(1)}

        def grade(schema, prompt):
            index = len(calls["prompts"])
            calls["prompts"].append(prompt)
            return grades[min(index, len(grades) - 1)]

        pipeline = load_pipeline(retrieve_documents=retrieve)
        pipeline._get_grader_model = lambda *_: FakeStructuredModel(grade)
        pipeline._get_complexity_model = lambda *_: FakeStructuredModel(
            lambda schema, prompt: {
                "complexity": "complex" if complex_question else "simple",
                "sub_questions": ["HMM initialisation", "HMM stability"]
                if complex_question
                else [],
            }
        )
        return pipeline, calls

    @staticmethod
    def grade(*, slot="工具包", answerability="partial", ambiguity="missing_slot"):
        return {
            "relevance": "strong",
            "answerability": answerability,
            "ambiguity": ambiguity,
            "route": "clarify",
            "missing_slots": [slot],
            "hitl_prompt": f"请补充{slot}",
        }

    def test_evidence_gaps_without_user_ambiguity_finish_after_one_rewrite(self):
        pipeline, calls = self.pipeline(
            [self.grade(slot="初始化稳定性的实验证据", ambiguity="none")]
        )
        result = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state("HMM要做多seed吗？", tenant_id="tenant-a")
        )
        self.assertEqual(2, len(calls["retrieve"]))
        self.assertEqual("answer", result["route"])
        self.assertEqual("partial", result["retrieval_status"])
        self.assertEqual("INSUFFICIENT_EVIDENCE", result["retrieval_outcome"])
        self.assertTrue(result["docs"])
        self.assertFalse(result["hitl_prompt"])

    def test_complex_subquestions_preserve_partial_evidence_despite_missing_details(
        self,
    ):
        pipeline, calls = self.pipeline([self.grade()], complex_question=True)
        result = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state(
                "请综合说明 HMM 的初始化方法及其稳定性评价", tenant_id="tenant-a"
            )
        )
        self.assertEqual(2, len(calls["retrieve"]))
        self.assertEqual("answer", result["route"])
        self.assertEqual("partial", result["retrieval_status"])
        self.assertTrue(result["docs"])

    def test_durable_resume_does_not_repeat_the_same_missing_slot(self):
        pipeline, _ = self.pipeline([self.grade()])
        paused = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state("这个工具的 HMM 怎么训练？", tenant_id="tenant-a")
        )
        self.assertEqual("clarify", paused["route"])
        resumed = pipeline.resume_rag_state(_compact_hitl_state(paused), "GLHMM")
        self.assertEqual("answer", resumed["route"])
        self.assertEqual("partial", resumed["retrieval_status"])
        self.assertTrue(resumed["docs"])
        self.assertEqual(
            "clarification_repeated", resumed["rag_trace"]["evidence_reason"]
        )

    def test_durable_resume_limits_even_changing_clarifications_to_two_rounds(self):
        pipeline, calls = self.pipeline(
            [self.grade(slot=slot) for slot in ("工具包", "版本", "稳定性指标")]
        )
        state = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state("这个工具的 HMM 怎么训练？", tenant_id="tenant-a")
        )
        state = pipeline.resume_rag_state(_compact_hitl_state(state), "GLHMM")
        self.assertEqual("clarify", state["route"])
        state = pipeline.resume_rag_state(_compact_hitl_state(state), "当前版本")
        self.assertEqual("answer", state["route"])
        self.assertEqual("partial", state["retrieval_status"])
        self.assertEqual(3, len(calls["retrieve"]))
        self.assertEqual(["GLHMM", "当前版本"], state["hitl_answers"])
        self.assertEqual(
            "clarification_limit_reached", state["rag_trace"]["evidence_reason"]
        )

    def test_declining_more_questions_returns_available_evidence(self):
        for answer in ("都可以，别问了", "看你能查到什么吧", "能展示的都展示"):
            with self.subTest(answer=answer):
                pipeline, _ = self.pipeline(
                    [self.grade(slot="工具包"), self.grade(slot="版本")]
                )
                state = pipeline.checkpointless_rag_graph.invoke(
                    pipeline._initial_state("HMM 的配置？", tenant_id="tenant-a")
                )
                state = pipeline.resume_rag_state(_compact_hitl_state(state), answer)
                self.assertEqual("answer", state["route"])
                self.assertEqual("partial", state["retrieval_status"])
                self.assertEqual(
                    "clarification_declined", state["rag_trace"]["evidence_reason"]
                )

    def test_no_answerable_evidence_after_clarification_ends_without_false_provider_failure(
        self,
    ):
        pipeline, _ = self.pipeline([self.grade(answerability="none")])
        state = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state("它的具体结论？", tenant_id="tenant-a")
        )
        state = pipeline.resume_rag_state(_compact_hitl_state(state), "GLHMM")
        self.assertEqual("insufficient_evidence", state["route"])
        self.assertEqual("INSUFFICIENT_EVIDENCE", state["retrieval_outcome"])
        self.assertFalse(state["docs"])
        ctx = RunRequestContext.for_sync(
            user_id="alice", thread_id="test", tenant_id="tenant-a"
        )
        try:
            rendered = _render_rag_result(ctx, state)
        finally:
            ctx.close()
        self.assertIn("INSUFFICIENT_EVIDENCE", rendered)
        self.assertNotIn("provider branches failed", rendered)
        self.assertNotIn("NEEDS_CLARIFICATION", rendered)

    def test_unresolved_scope_is_not_reported_as_a_complete_answer_at_limit(self):
        pipeline, _ = self.pipeline(
            [self.grade(answerability="sufficient", ambiguity="multiple_candidates")]
        )
        state = pipeline._initial_state("这个型号的参数？", tenant_id="tenant-a")
        state["hitl_answers"] = ["产品线 A", "当前版本"]
        state = pipeline.resume_rag_state(_compact_hitl_state(state), "生产型号")
        self.assertEqual("answer", state["route"])
        self.assertEqual("partial", state["retrieval_status"])
        self.assertEqual("INSUFFICIENT_EVIDENCE", state["retrieval_outcome"])

    def test_related_but_unanswerable_complex_evidence_is_not_no_knowledge(self):
        pipeline, _ = self.pipeline(
            [self.grade(answerability="none", ambiguity="none")], complex_question=True
        )
        state = pipeline.checkpointless_rag_graph.invoke(
            pipeline._initial_state(
                "请说明 HMM 初始化及稳定性的量化结果", tenant_id="tenant-a"
            )
        )
        self.assertEqual("insufficient_evidence", state["route"])
        self.assertEqual("INSUFFICIENT_EVIDENCE", state["retrieval_outcome"])
        self.assertFalse(state["docs"])

    def test_no_evidence_with_a_provider_failure_still_discloses_the_failure(self):
        ctx = RunRequestContext.for_sync(
            user_id="alice", thread_id="test", tenant_id="tenant-a"
        )
        try:
            rendered = _render_rag_result(
                ctx,
                {
                    "docs": [],
                    "rag_trace": {
                        "retrieval_outcome": "INSUFFICIENT_EVIDENCE",
                        "coverage_gap_codes": ["VECTOR_STORE_UNAVAILABLE"],
                    },
                },
            )
        finally:
            ctx.close()
        self.assertIn("provider branches failed", rendered)
        self.assertIn("VECTOR_STORE_UNAVAILABLE", rendered)
