from __future__ import annotations

from backend.core.settings import AppSettings, get_settings
from backend.evaluation.contracts import (
    RagEvaluationCaseRecord,
    RagEvaluationDatasetRecord,
    RagEvaluationJobRecord,
    RagEvaluationJobStatus,
)
from backend.evaluation.rag import RagEvalDataset, RagEvalGatePolicy, RagMetricGate
from backend.evaluation.repository import (
    RagEvaluationRepository,
    rag_evaluation_repository,
)
from backend.model_control import ModelControlService, ModelRole, model_control_service
from backend.core.errors import AppError, ErrorCode
from backend.documents.retrieval import DocumentRetrievalScope
from backend.evaluation.experiments import (
    RagExperimentRequest,
    RagExperimentSnapshot,
    experiment_runtime_fingerprint,
)
from backend.rag.profiles import RagRetrievalProfile


DEFAULT_RAG_EVALUATION_GATES = RagEvalGatePolicy(
    k_values=(5, 10),
    critical_no_regression=True,
    required_provenance="live_rag",
    metric_gates=(RagMetricGate(metric="case_pass_rate", minimum=0.95),),
)


class RagEvaluationService:
    """Application Interface for persistent RAG evaluation control-plane actions."""

    def __init__(
        self,
        repository: RagEvaluationRepository = rag_evaluation_repository,
        *,
        model_control: ModelControlService = model_control_service,
        settings: AppSettings | None = None,
        retrieval_scope: DocumentRetrievalScope | None = None,
    ) -> None:
        self.repository = repository
        self.model_control = model_control
        self.settings = settings or get_settings()
        self.retrieval_scope = retrieval_scope

    def create_dataset(
        self,
        *,
        username: str,
        dataset: RagEvalDataset,
    ) -> RagEvaluationDatasetRecord:
        return self.repository.create_dataset(username=username, dataset=dataset)

    def list_datasets(
        self, *, limit: int = 100
    ) -> tuple[RagEvaluationDatasetRecord, ...]:
        return self.repository.list_datasets(limit=limit)

    def get_dataset(self, dataset_id: str) -> RagEvaluationDatasetRecord:
        return self.repository.get_dataset(dataset_id)

    def create_job(
        self,
        *,
        username: str,
        dataset_id: str,
        gate_policy: RagEvalGatePolicy | None = None,
        baseline_job_id: str | None = None,
        experiment: RagExperimentRequest | None = None,
    ) -> RagEvaluationJobRecord:
        if experiment is not None and baseline_job_id is not None:
            raise AppError(
                ErrorCode.CONFLICT,
                "对比实验请使用实验报告汇总，不与发布 baseline 混用",
                status_code=409,
                category="evaluation",
                stage="experiment",
            )
        experiment_snapshot = (
            self._freeze_experiment(experiment) if experiment else None
        )
        model_snapshot = self.model_control.runtime_snapshot(
            required_roles=frozenset(ModelRole)
        )
        return self.repository.create_job(
            username=username,
            dataset_id=dataset_id,
            gate_policy=gate_policy or DEFAULT_RAG_EVALUATION_GATES,
            model_snapshot=model_snapshot,
            baseline_job_id=baseline_job_id,
            max_attempts=self.settings.worker.evaluation_max_attempts,
            experiment=experiment_snapshot,
        )

    def _freeze_experiment(
        self, request: RagExperimentRequest
    ) -> RagExperimentSnapshot:
        from backend.evaluation.experiments import experiment_definition

        definition = experiment_definition(request.name)
        if definition.rerank_enabled and not self.settings.rerank.enabled:
            raise AppError(
                ErrorCode.CONFLICT,
                "重排实验需要配置可用的 Reranker",
                status_code=409,
                category="evaluation",
                stage="experiment",
            )
        scope = self.retrieval_scope or DocumentRetrievalScope()
        snapshot = scope.resolve(
            tenant_id=self.settings.app.default_tenant_id,
            knowledge_base_id=request.knowledge_base_id,
        )
        if (
            not snapshot.targets
            or snapshot.current_document_count != snapshot.catalog_document_count
        ):
            raise AppError(
                ErrorCode.CONFLICT,
                "评测知识库为空或仍有未就绪文档",
                status_code=409,
                category="evaluation",
                stage="experiment",
            )
        from backend.rag.utils import AUTO_MERGE_ENABLED, AUTO_MERGE_THRESHOLD

        return RagExperimentSnapshot(
            name=request.name,
            tenant_id=snapshot.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            corpus_fingerprint=request.corpus_fingerprint,
            index_id=snapshot.index_id,
            document_count=snapshot.current_document_count,
            retrieval=RagRetrievalProfile(
                mode=definition.mode,
                rerank_enabled=definition.rerank_enabled,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                auto_merge_enabled=AUTO_MERGE_ENABLED,
                auto_merge_threshold=AUTO_MERGE_THRESHOLD,
            ),
            runtime_fingerprint=experiment_runtime_fingerprint(),
        )

    def create_comparison_jobs(
        self,
        *,
        username: str,
        dataset_id: str,
        experiments: tuple[RagExperimentRequest, ...],
    ) -> tuple[RagEvaluationJobRecord, ...]:
        from backend.evaluation.experiments import comparison_design

        comparison_design([item.name for item in experiments])
        snapshots = tuple(self._freeze_experiment(request) for request in experiments)
        if (
            len(
                {
                    (
                        item.tenant_id,
                        item.knowledge_base_id,
                        item.index_id,
                        item.corpus_fingerprint,
                        item.document_count,
                        item.runtime_fingerprint,
                        item.retrieval.top_k,
                        item.retrieval.candidate_k,
                        item.retrieval.auto_merge_enabled,
                        item.retrieval.auto_merge_threshold,
                    )
                    for item in snapshots
                }
            )
            != 1
        ):
            raise ValueError("comparison scope changed during preflight")
        model_snapshot = self.model_control.runtime_snapshot(
            required_roles=frozenset(ModelRole)
        )
        # Experiments report measurements; they do not relax or replace release gates.
        policy = RagEvalGatePolicy(required_provenance="live_rag", k_values=(5, 10))
        return self.repository.create_comparison_jobs(
            username=username,
            dataset_id=dataset_id,
            gate_policy=policy,
            model_snapshot=model_snapshot,
            experiments=snapshots,
            max_attempts=self.settings.worker.evaluation_max_attempts,
        )

    def list_jobs(
        self,
        *,
        status: RagEvaluationJobStatus | str | None = None,
        limit: int = 100,
    ) -> tuple[RagEvaluationJobRecord, ...]:
        return self.repository.list_jobs(status=status, limit=limit)

    def get_job(self, job_id: str) -> RagEvaluationJobRecord:
        return self.repository.get_job(job_id)

    def list_cases(self, job_id: str) -> tuple[RagEvaluationCaseRecord, ...]:
        return self.repository.list_cases(job_id)

    def cancel_job(self, job_id: str) -> RagEvaluationJobRecord:
        return self.repository.cancel(job_id=job_id)


rag_evaluation_service = RagEvaluationService()


__all__ = [
    "DEFAULT_RAG_EVALUATION_GATES",
    "RagEvaluationService",
    "rag_evaluation_service",
]
