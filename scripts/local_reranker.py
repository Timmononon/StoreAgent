"""Serve a cached GPU reranker through the existing /v1/rerank protocol."""

from __future__ import annotations

import asyncio
import hmac
import math
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_ALIAS = f"{MODEL_ID}@{MODEL_REVISION}"


@dataclass(frozen=True)
class LocalRerankerConfig:
    model_path: Path
    api_key: str = field(repr=False)
    batch_size: int = 4
    max_length: int = 1024

    def __post_init__(self) -> None:
        if not self.api_key or not 1 <= self.batch_size <= 16:
            raise ValueError("a local API key and batch size 1–16 are required")
        if not 128 <= self.max_length <= 2048:
            raise ValueError("max_length must be between 128 and 2048")


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=2000)
    documents: list[Annotated[str, Field(min_length=1, max_length=8000)]] = Field(
        min_length=1, max_length=30
    )
    top_n: int = Field(ge=1, le=30)
    return_documents: Literal[False] = False

    @model_validator(mode="after")
    def bound_total_input(self) -> "RerankRequest":
        if sum(map(len, self.documents)) > 60000:
            raise ValueError("document character budget exceeded")
        return self


class Scorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...

    def close(self) -> None: ...


class GpuScorer:
    def __init__(self, config: LocalRerankerConfig) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("local reranker requires CUDA")
        self._torch = torch
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(config.model_path),
            dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).to("cuda")
        self._model.eval()
        self.score("What is a panda?", ["A panda is a bear native to China."])

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(documents), self._config.batch_size):
                pairs = [
                    [query, text]
                    for text in documents[start : start + self._config.batch_size]
                ]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_length,
                    return_tensors="pt",
                ).to("cuda")
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(self._torch.sigmoid(logits).cpu().tolist())
        return scores

    def close(self) -> None:
        del self._model
        self._torch.cuda.empty_cache()


def create_app(
    config: LocalRerankerConfig | None = None,
    *,
    scorer_factory: Callable[[LocalRerankerConfig], Scorer] = GpuScorer,
) -> FastAPI:
    if config is None:
        from backend.env import load_env

        load_env()
        config = LocalRerankerConfig(
            model_path=Path(os.environ["LOCAL_RERANK_MODEL_PATH"]),
            api_key=os.environ["RERANK_API_KEY"],
            batch_size=int(os.getenv("LOCAL_RERANK_BATCH_SIZE", "4")),
            max_length=int(os.getenv("LOCAL_RERANK_MAX_LENGTH", "1024")),
        )
    settings = config

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        scorer = await run_in_threadpool(scorer_factory, settings)
        app.state.scorer = scorer
        app.state.gate = asyncio.Lock()
        app.state.requests_completed = 0
        try:
            yield
        finally:
            await run_in_threadpool(scorer.close)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ready": True,
            "model": MODEL_ALIAS,
            "device": "cuda",
            "dtype": "float16",
            "batch_size": settings.batch_size,
            "max_length": settings.max_length,
            "requests_completed": app.state.requests_completed,
        }

    @app.post("/v1/rerank")
    async def rerank(
        body: RerankRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict:
        if not hmac.compare_digest(authorization or "", f"Bearer {settings.api_key}"):
            raise HTTPException(401, "RERANK_UNAUTHORIZED")
        if body.model != MODEL_ALIAS:
            raise HTTPException(400, "RERANK_MODEL_MISMATCH")
        async with app.state.gate:
            try:
                scores = await run_in_threadpool(
                    app.state.scorer.score, body.query, body.documents
                )
            except Exception:
                raise HTTPException(503, "RERANK_INFERENCE_FAILED") from None
            if len(scores) != len(body.documents) or any(
                not math.isfinite(score) for score in scores
            ):
                raise HTTPException(503, "RERANK_INVALID_SCORES")
            order = sorted(
                range(len(scores)), key=lambda index: (-scores[index], index)
            )
            app.state.requests_completed += 1
            return {
                "results": [
                    {"index": index, "relevance_score": scores[index]}
                    for index in order[: body.top_n]
                ]
            }

    return app
