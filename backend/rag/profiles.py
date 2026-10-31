"""Immutable retrieval parameters shared by production RAG and experiments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RagSynthesisStrategy = Literal["concatenate_v1", "source_priority_v1"]


class RagRetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["dense", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=30, ge=1, le=500)
    auto_merge_enabled: bool = True
    auto_merge_threshold: int = Field(default=2, ge=1, le=100)

    @model_validator(mode="after")
    def validate_candidates(self) -> "RagRetrievalProfile":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self
