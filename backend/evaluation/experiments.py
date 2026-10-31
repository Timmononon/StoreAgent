"""Frozen, secret-free execution identity for a benchmark Evaluation Job."""

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.rag.profiles import RagRetrievalProfile, RagSynthesisStrategy


RagExperimentName = Literal[
    "dense",
    "hybrid",
    "hybrid_rerank",
    "dense_baseline",
    "hybrid_baseline",
    "hybrid_rerank_baseline",
    "hybrid_rerank_optimized",
]
RagComparisonDesign = Literal["retrieval", "optimization"]


class RagExperimentDefinition(NamedTuple):
    mode: Literal["dense", "hybrid"]
    rerank_enabled: bool
    synthesis_strategy: RagSynthesisStrategy
    label: str


def experiment_definition(name: RagExperimentName) -> RagExperimentDefinition:
    """Immutable experiment identities; historical names keep their semantics."""
    match name:
        case "dense":
            return RagExperimentDefinition(
                "dense", False, "source_priority_v1", "Dense"
            )
        case "hybrid":
            return RagExperimentDefinition(
                "hybrid", False, "source_priority_v1", "Hybrid"
            )
        case "hybrid_rerank":
            return RagExperimentDefinition(
                "hybrid", True, "source_priority_v1", "Hybrid + Reranker"
            )
        case "dense_baseline":
            return RagExperimentDefinition("dense", False, "concatenate_v1", "A: Dense")
        case "hybrid_baseline":
            return RagExperimentDefinition(
                "hybrid", False, "concatenate_v1", "B: Hybrid"
            )
        case "hybrid_rerank_baseline":
            return RagExperimentDefinition(
                "hybrid", True, "concatenate_v1", "C0: Hybrid + Reranker"
            )
        case "hybrid_rerank_optimized":
            return RagExperimentDefinition(
                "hybrid", True, "source_priority_v1", "C1: Optimized Hybrid + Reranker"
            )
        case _:
            raise ValueError("unknown experiment identity")


def comparison_names(design: RagComparisonDesign) -> tuple[RagExperimentName, ...]:
    if design == "retrieval":
        return ("dense", "hybrid", "hybrid_rerank")
    if design == "optimization":
        return (
            "dense_baseline",
            "hybrid_baseline",
            "hybrid_rerank_baseline",
            "hybrid_rerank_optimized",
        )
    raise ValueError("unknown comparison design")


def comparison_design(names: Sequence[str]) -> RagComparisonDesign:
    for design in ("retrieval", "optimization"):
        expected = comparison_names(design)
        if len(names) == len(expected) and set(names) == set(expected):
            return design
    raise ValueError(
        "comparison requires three retrieval arms or four optimization arms"
    )


class RagExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: RagExperimentName
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=30, ge=1, le=500)

    @model_validator(mode="after")
    def validate_candidates(self) -> "RagExperimentRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self


class RagExperimentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: RagExperimentName
    tenant_id: str = Field(min_length=1, max_length=64)
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=1)
    retrieval: RagRetrievalProfile
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_profile(self) -> "RagExperimentSnapshot":
        definition = experiment_definition(self.name)
        if self.retrieval.mode != definition.mode:
            raise ValueError("experiment name and retrieval mode disagree")
        if self.retrieval.rerank_enabled != definition.rerank_enabled:
            raise ValueError("experiment name and rerank configuration disagree")
        return self

    @property
    def synthesis_strategy(self) -> RagSynthesisStrategy:
        return experiment_definition(self.name).synthesis_strategy


def experiment_runtime_fingerprint() -> str:
    """Pin effective non-secret provider/settings values across worker restarts."""
    from backend.core.settings import get_settings

    settings = get_settings()
    payload = {
        "rag": settings.rag.model_dump(mode="json"),
        "embedding": settings.embedding.model_dump(mode="json"),
        "rerank": settings.rerank.model_dump(
            mode="json", exclude={"api_key", "binding_host"}
        ),
        # Endpoints affect comparability but must never be published verbatim.
        "rerank_endpoint_hash": hashlib.sha256(
            settings.rerank.binding_host.encode()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
