from __future__ import annotations

from types import SimpleNamespace
import hashlib

import pytest
from sqlalchemy import create_engine, inspect
from alembic import command

import backend.evaluation.runtime as runtime_module
import backend.evaluation.service as service_module
import backend.evaluation.worker as worker_module
from backend.core.errors import AppError
from backend.documents.retrieval import RetrievalSnapshot, RetrievalTarget
from backend.evaluation.benchmark import (
    compare_jobs,
    enqueue_comparison,
    render_comparison,
)
from backend.evaluation.crud import prepare_crud
from backend.evaluation.rag import load_rag_eval_dataset
from backend.evaluation.experiments import (
    RagExperimentSnapshot,
    comparison_names,
    experiment_definition,
)
from backend.evaluation.rag import RagEvalGatePolicy
from backend.rag.profiles import RagRetrievalProfile
from backend.evaluation.runtime import RagEvaluationRuntime
from backend.evaluation.worker import RagEvaluationWorker
from backend.infra.database import alembic_config
from backend.providers import ProviderExecutor
from test_crud_rag_benchmark import make_source
from test_rag_evaluation_jobs import _dataset, _environment, _settings
from test_rag_evaluation_judge_runtime import _Models
from tests.support import TEST_MODEL_SNAPSHOT


@pytest.mark.parametrize("design", ["retrieval", "optimization"])
def test_100_case_worker_flow_is_durable_and_comparable(tmp_path, monkeypatch, design):
    source = make_source(tmp_path / "source")
    directory = tmp_path / "prepared"
    manifest = prepare_crud(source, directory)
    dataset = load_rag_eval_dataset(directory / "dataset.json")
    engine, _, repository, service = _environment()
    snapshot = RetrievalSnapshot(
        tenant_id="test",
        index_id="a" * 64,
        targets=(
            RetrievalTarget(collection_name="benchmark", filter_expr="version-filter"),
        ),
        current_document_count=420,
        catalog_document_count=420,
    )
    scope = SimpleNamespace(resolve=lambda **_: snapshot)
    service.retrieval_scope = scope
    service.settings = SimpleNamespace(
        app=SimpleNamespace(default_tenant_id="test"),
        rerank=SimpleNamespace(enabled=True),
        worker=_settings().worker,
    )
    monkeypatch.setattr(
        service_module, "experiment_runtime_fingerprint", lambda: "b" * 64
    )
    monkeypatch.setattr(
        worker_module, "experiment_runtime_fingerprint", lambda: "b" * 64
    )
    monkeypatch.setattr(worker_module, "DocumentRetrievalScope", lambda: scope)
    questions = {case.question: case for case in dataset.cases}
    observed = []

    def graph(question, context):
        profile = context.rag_retrieval_profile
        observed.append(
            (
                profile.mode,
                profile.rerank_enabled,
                context.tenant_id,
                context.rag_synthesis_strategy,
            )
        )
        assert context.get_or_resolve_rag_retrieval_snapshot(lambda: None) is snapshot
        case = questions[question]
        documents = [
            {
                "filename": gold.canonical_name,
                "chunk_id": f"{case.id}:{index}",
                "document_id": gold.canonical_name,
                "document_version_id": "version",
                "content_hash": hashlib.sha256(
                    gold.canonical_name.encode()
                ).hexdigest(),
                "index_version": snapshot.index_id,
                "text": f"controlled fixture evidence {gold.canonical_name}",
            }
            for index, gold in enumerate(case.gold_documents)
        ]
        return {
            "docs": documents,
            "route": "answer",
            "retrieval_outcome": "ANSWERABLE",
            "rag_trace": {
                "retrieved_chunks": documents,
                "retrieval_index_id": snapshot.index_id,
                "route": "answer",
                "retrieval_outcome": "ANSWERABLE",
            },
        }

    monkeypatch.setattr(runtime_module, "run_rag_graph", graph)
    models = _Models()
    runtime = RagEvaluationRuntime(
        models=models, executor=ProviderExecutor(sleeper=lambda _: None)
    )
    try:
        record = service.create_dataset(username="admin", dataset=dataset)
        jobs = enqueue_comparison(
            service,
            username="admin",
            dataset_id=record.id,
            knowledge_base_id="kb",
            corpus_fingerprint=manifest["corpus_fingerprint"],
            design=design,
        )
        assert len({job.dataset_fingerprint for job in jobs}) == 1
        assert len({job.model_catalog_hash for job in jobs}) == 1
        assert len({job.experiment.index_id for job in jobs}) == 1
        worker = RagEvaluationWorker(
            repository=repository, runtime=runtime, settings=_settings()
        )
        for _ in jobs:
            assert worker.run_once()
        finished = [repository.get_job(job.id) for job in jobs]
        report = compare_jobs(finished)
        assert report["valid"]
        assert report["case_count"] == 100
        names = comparison_names(design)
        assert len(observed) == 100 * len(names)
        assert set(observed) == {
            (
                definition.mode,
                definition.rerank_enabled,
                "test",
                definition.synthesis_strategy,
            )
            for definition in (experiment_definition(name) for name in names)
        }
        assert (
            report["experiments"][names[0]]["metrics"]["document_recall_at_10"]["value"]
            == 1
        )
        assert all(job.completed_cases == 100 for job in finished)
        assert "rag_latency_p95_ms" in report["experiments"][names[0]]["metrics"]
        assert "Hybrid + Reranker" in render_comparison(report)
        if design == "retrieval":
            # Historical names did not freeze a synthesis strategy separately;
            # its old report must not be relabeled with the current algorithm.
            assert all(
                arm["synthesis_strategy"] is None
                for arm in report["experiments"].values()
            )
        if design == "optimization":
            assert (
                report["experiments"][names[2]]["synthesis_strategy"]
                == "concatenate_v1"
            )
            assert (
                report["experiments"][names[3]]["synthesis_strategy"]
                == "source_priority_v1"
            )
            assert (
                report["contrasts"]["C1 vs C0"]["metrics"]["document_recall_at_10"][
                    "absolute_delta"
                ]
                == 0
            )
            assert "C0 vs A" in render_comparison(report)
            assert "C1 vs C0" in render_comparison(report)
        metrics = {**finished[0].report.metrics}
        metrics["retrieval_degraded"] = metrics["retrieval_degraded"].model_copy(
            update={"value": 0.01}
        )
        degraded_report = finished[0].report.model_copy(update={"metrics": metrics})
        degraded = compare_jobs(
            [finished[0].model_copy(update={"report": degraded_report}), *finished[1:]]
        )
        assert not degraded["valid"]
        assert degraded["invalid_reasons"] == [
            f"{finished[0].experiment.name}:retrieval_degraded"
        ]
        metrics = {
            key: value for key, value in metrics.items() if key != "retrieval_degraded"
        }
        incomplete_report = finished[0].report.model_copy(update={"metrics": metrics})
        incomplete = compare_jobs(
            [
                finished[0].model_copy(update={"report": incomplete_report}),
                *finished[1:],
            ]
        )
        assert not incomplete["valid"]
        assert incomplete["invalid_reasons"] == [
            f"{finished[0].experiment.name}:retrieval_degraded_incomplete"
        ]
        with pytest.raises(AppError, match="Baseline"):
            service.create_job(
                username="admin", dataset_id=record.id, baseline_job_id=finished[0].id
            )
        altered_report = finished[0].report.model_copy(
            update={
                "metadata": {
                    **finished[0].report.metadata,
                    "model_catalog_hash": "c" * 64,
                }
            }
        )
        altered = finished[0].model_copy(
            update={"model_catalog_hash": "c" * 64, "report": altered_report}
        )
        with pytest.raises(ValueError, match="differ"):
            compare_jobs([altered, *finished[1:]])
        missing_source = finished[0].report.model_copy(update={"metadata": {}})
        with pytest.raises(ValueError):
            compare_jobs(
                [
                    finished[0].model_copy(update={"report": missing_source}),
                    *finished[1:],
                ]
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize("design", ["retrieval", "optimization"])
def test_comparison_refuses_missing_reranker_before_queuing_any_job(design):
    engine, _, repository, service = _environment()
    service.settings = SimpleNamespace(
        app=SimpleNamespace(default_tenant_id="test"),
        rerank=SimpleNamespace(enabled=False),
    )
    service.retrieval_scope = SimpleNamespace(
        resolve=lambda **_: RetrievalSnapshot(
            tenant_id="test",
            index_id="a" * 64,
            targets=(RetrievalTarget(collection_name="test", filter_expr="versions"),),
            current_document_count=1,
            catalog_document_count=1,
        )
    )
    try:
        with pytest.raises(AppError, match="Reranker"):
            enqueue_comparison(
                service,
                username="admin",
                dataset_id="unused",
                knowledge_base_id="kb",
                corpus_fingerprint="f" * 64,
                design=design,
            )
        assert not repository.list_jobs()
    finally:
        engine.dispose()


def test_migration_adds_nullable_experiment(tmp_path):
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    config = alembic_config(url)
    command.upgrade(config, "0018_capability_control_plane")
    engine = create_engine(url)
    assert "experiment_json" not in {
        column["name"] for column in inspect(engine).get_columns("rag_evaluation_jobs")
    }
    engine.dispose()
    command.upgrade(config, "head")
    engine = create_engine(url)
    column = next(
        column
        for column in inspect(engine).get_columns("rag_evaluation_jobs")
        if column["name"] == "experiment_json"
    )
    assert column["nullable"]
    engine.dispose()


def _experiment(name="dense"):
    definition = experiment_definition(name)
    return RagExperimentSnapshot(
        name=name,
        tenant_id="test",
        knowledge_base_id="kb",
        corpus_fingerprint="f" * 64,
        index_id="a" * 64,
        document_count=1,
        runtime_fingerprint="b" * 64,
        retrieval=RagRetrievalProfile(
            mode=definition.mode,
            rerank_enabled=definition.rerank_enabled,
        ),
    )


@pytest.mark.parametrize("design", ["retrieval", "optimization"])
def test_failed_cohort_submission_rolls_back_all_arms(monkeypatch, design):
    engine, _, repository, service = _environment()
    record = service.create_dataset(username="admin", dataset=_dataset())
    original = repository.create_job

    def create(**kwargs):
        if kwargs["experiment"].name == comparison_names(design)[-1]:
            raise RuntimeError("injected third insertion failure")
        return original(**kwargs)

    monkeypatch.setattr(repository, "create_job", create)
    try:
        with pytest.raises(RuntimeError, match="third insertion"):
            repository.create_comparison_jobs(
                username="admin",
                dataset_id=record.id,
                gate_policy=RagEvalGatePolicy(),
                model_snapshot=TEST_MODEL_SNAPSHOT,
                max_attempts=1,
                experiments=tuple(
                    _experiment(name) for name in comparison_names(design)
                ),
            )
        assert repository.list_jobs() == ()
    finally:
        engine.dispose()


def test_index_drift_fails_before_any_case_or_model_call(monkeypatch):
    engine, _, repository, service = _environment()
    try:
        record = service.create_dataset(username="admin", dataset=_dataset())
        job = repository.create_job(
            username="admin",
            dataset_id=record.id,
            gate_policy=RagEvalGatePolicy(),
            model_snapshot=TEST_MODEL_SNAPSHOT,
            experiment=_experiment(),
        )
        monkeypatch.setattr(
            worker_module, "experiment_runtime_fingerprint", lambda: "b" * 64
        )
        changed = SimpleNamespace(index_id="c" * 64, current_document_count=1)
        monkeypatch.setattr(
            worker_module,
            "DocumentRetrievalScope",
            lambda: SimpleNamespace(resolve=lambda **_: changed),
        )
        worker = RagEvaluationWorker(
            repository=repository, runtime=SimpleNamespace(), settings=_settings()
        )
        assert worker.run_once()
        failed = repository.get_job(job.id)
        assert failed.status.value == "failed"
        assert failed.error_code == "RAG_EVALUATION_INDEX_CHANGED"
        assert failed.completed_cases == 0
    finally:
        engine.dispose()
