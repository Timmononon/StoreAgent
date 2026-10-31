"""Benchmark orchestration over the existing Dataset/Job and publication seams."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.evaluation.contracts import RagEvaluationJobRecord, RagEvaluationJobStatus
from backend.evaluation.experiments import (
    RagComparisonDesign,
    RagExperimentRequest,
    comparison_design,
    comparison_names,
    experiment_definition,
)
from backend.evaluation.service import RagEvaluationService


EXPERIMENT_NAMES = comparison_names("retrieval")
REPORTED_METRICS = (
    "document_recall_at_10",
    "all_documents_hit_at_10",
    "answer_correctness",
    "groundedness",
    "completeness",
    "rag_latency_p95_ms",
    "retrieval_degraded",
    "provider_failure_rate",
)


def enqueue_comparison(
    service: RagEvaluationService,
    *,
    username: str,
    dataset_id: str,
    knowledge_base_id: str,
    corpus_fingerprint: str,
    design: RagComparisonDesign = "retrieval",
) -> tuple[RagEvaluationJobRecord, ...]:
    """Preflight every arm before atomically submitting the frozen cohort."""
    requests = tuple(
        RagExperimentRequest(
            name=name,
            knowledge_base_id=knowledge_base_id,
            corpus_fingerprint=corpus_fingerprint,
        )
        for name in comparison_names(design)
    )
    return service.create_comparison_jobs(
        username=username, dataset_id=dataset_id, experiments=requests
    )


def compare_jobs(jobs: Sequence[RagEvaluationJobRecord]) -> dict[str, Any]:
    if not jobs or any(job.experiment is None for job in jobs):
        raise ValueError("comparison requires frozen experiment Jobs")
    by_name = {job.experiment.name: job for job in jobs if job.experiment}
    design = comparison_design([job.experiment.name for job in jobs if job.experiment])
    names = comparison_names(design)
    first = jobs[0]
    first_report = first.report
    if first_report is None:
        raise ValueError("all experiment Jobs must finish with persisted reports")
    identity = None
    case_ids = None
    invalid: list[str] = []
    for job in jobs:
        experiment = job.experiment
        if (
            experiment is None
            or job.status is not RagEvaluationJobStatus.SUCCEEDED
            or job.report is None
        ):
            raise ValueError("all experiment Jobs must finish with persisted reports")
        current = (
            job.dataset_fingerprint,
            job.model_catalog_hash,
            job.total_cases,
            experiment.tenant_id,
            experiment.knowledge_base_id,
            experiment.index_id,
            experiment.corpus_fingerprint,
            experiment.document_count,
            experiment.runtime_fingerprint,
            experiment.retrieval.top_k,
            experiment.retrieval.candidate_k,
            experiment.retrieval.auto_merge_enabled,
            experiment.retrieval.auto_merge_threshold,
            job.report.metadata.get("rag_source_fingerprint"),
        )
        if identity is not None and current != identity:
            raise ValueError(
                "experiment datasets, models, corpus, index or fixed parameters differ"
            )
        identity = current
        if job.report.metadata.get("provenance") != "live_rag":
            raise ValueError("comparison only accepts live_rag worker reports")
        if (
            not job.report.metadata.get("rag_source_fingerprint")
            or job.report.dataset_fingerprint != job.dataset_fingerprint
            or job.report.metadata.get("model_catalog_hash") != job.model_catalog_hash
            or job.report.metadata.get("experiment")
            != experiment.model_dump(mode="json")
            or job.report.case_count != job.total_cases
        ):
            raise ValueError("report identity does not match its frozen Job")
        current_ids = {case.case_id for case in job.report.cases}
        if len(current_ids) != job.total_cases or (
            case_ids is not None and current_ids != case_ids
        ):
            raise ValueError("comparison contains different case identities")
        case_ids = current_ids
        if (
            job.completed_cases != job.total_cases
            or job.report.observation_count != job.total_cases
        ):
            raise ValueError("comparison contains incomplete observations")
        for name in ("provider_failure_rate", "retrieval_degraded"):
            metric = job.report.metrics.get(name)
            if (
                metric is None
                or metric.value is None
                or metric.eligible_cases != job.total_cases
            ):
                invalid.append(f"{experiment.name}:{name}_incomplete")
            elif metric.value > 0:
                invalid.append(f"{experiment.name}:{name}")
    contrasts = {}
    if design == "optimization":
        for label, reference, candidate in (
            ("B vs A", names[0], names[1]),
            ("C0 vs A", names[0], names[2]),
            ("C0 vs B", names[1], names[2]),
            ("C1 vs C0", names[2], names[3]),
            ("C1 vs A", names[0], names[3]),
        ):
            before = by_name[reference].report
            after = by_name[candidate].report
            assert before is not None and after is not None
            changes = {}
            for metric in REPORTED_METRICS:
                a, b = before.metrics.get(metric), after.metrics.get(metric)
                delta = (
                    b.value - a.value
                    if a and b and a.value is not None and b.value is not None
                    else None
                )
                changes[metric] = {
                    "absolute_delta": delta,
                    "relative_change": delta / abs(a.value)
                    if delta is not None and a and a.value
                    else None,
                    "reference_eligible_cases": a.eligible_cases if a else 0,
                    "candidate_eligible_cases": b.eligible_cases if b else 0,
                }
            contrasts[label] = {
                "reference": reference,
                "candidate": candidate,
                "metrics": changes,
            }
    return {
        "schema_version": 1,
        "kind": "controlled_benchmark_comparison",
        "comparison_design": design,
        "experiment_order": names,
        "contrasts": contrasts,
        "release_gate": False,
        "valid": not invalid,
        "invalid_reasons": invalid,
        "dataset_fingerprint": first.dataset_fingerprint,
        "dataset_name": first.dataset_name,
        "case_count": first.total_cases,
        "model_catalog_hash": first.model_catalog_hash,
        "rag_source_fingerprint": first_report.metadata["rag_source_fingerprint"],
        "experiments": {
            name: {
                "job_id": job.id,
                "profile": job.experiment.model_dump(mode="json")
                if job.experiment
                else None,
                "synthesis_strategy": job.experiment.synthesis_strategy
                if job.experiment and design == "optimization"
                else None,
                "metrics": {
                    key: value.model_dump(mode="json")
                    for key, value in job.report.metrics.items()
                },
                "slices": {
                    key: value.model_dump(mode="json")
                    for key, value in job.report.slices.items()
                },
            }
            for name, job in sorted(by_name.items())
            if job.report
        },
    }


def render_comparison(report: dict[str, Any]) -> str:
    names = comparison_names(report["comparison_design"])
    lines = [
        f"# {report['dataset_name']} controlled comparison",
        "",
        f"Cases: {report['case_count']}; valid controls: {report['valid']}",
        "",
        "| Metric | "
        + " | ".join(experiment_definition(name).label for name in names)
        + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for metric in REPORTED_METRICS:
        cells = []
        for name in names:
            result = report["experiments"][name]["metrics"].get(metric, {})
            value = result.get("value")
            cells.append(
                "unavailable"
                if value is None
                else f"{value:.4f} (n={result['eligible_cases']})"
            )
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    for label, contrast in report["contrasts"].items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "Signed change: candidate minus reference; lower latency/failure is better.",
                "",
                "| Metric | Absolute delta | Relative change | n reference/candidate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric, change in contrast["metrics"].items():
            delta = change["absolute_delta"]
            relative = change["relative_change"]
            delta_text = "unavailable" if delta is None else f"{delta:+.4f}"
            relative_text = "unavailable" if relative is None else f"{relative:+.2%}"
            lines.append(
                f"| {metric} | {delta_text} | {relative_text} | {change['reference_eligible_cases']}/{change['candidate_eligible_cases']} |"
            )
    if report["invalid_reasons"]:
        lines.extend(["", "Invalid controls: " + ", ".join(report["invalid_reasons"])])
    return "\n".join(lines) + "\n"
