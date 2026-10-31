from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.local_reranker import MODEL_ALIAS, LocalRerankerConfig, create_app


class FakeScorer:
    def __init__(self, _config):
        self.closed = False
        self.calls = 0

    def score(self, _query, documents):
        self.calls += 1
        return [0.1, 0.9, 0.9][: len(documents)]

    def close(self):
        self.closed = True


def test_authenticated_protocol_ranks_original_indexes_and_closes_model():
    scorer = FakeScorer(None)
    app = create_app(
        LocalRerankerConfig(Path("unused"), "local-test-key"),
        scorer_factory=lambda _: scorer,
    )
    with TestClient(app) as client:
        payload = {
            "model": MODEL_ALIAS,
            "query": "question",
            "documents": ["first", "second", "third"],
            "top_n": 2,
            "return_documents": False,
        }
        assert client.post("/v1/rerank", json=payload).status_code == 401
        assert scorer.calls == 0
        response = client.post(
            "/v1/rerank",
            json=payload,
            headers={"Authorization": "Bearer local-test-key"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.9},
            ]
        }
        assert client.get("/health").json()["requests_completed"] == 1
    assert scorer.closed


@pytest.mark.parametrize(
    "change",
    [
        {"documents": []},
        {"documents": ["x"] * 31},
        {"documents": ["x" * 8001]},
        {"documents": ["x" * 8000] * 8},
        {"return_documents": True},
        {"top_n": 0},
    ],
)
def test_input_budgets_reject_before_model_inference(change):
    scorer = FakeScorer(None)
    app = create_app(
        LocalRerankerConfig(Path("unused"), "local-test-key"),
        scorer_factory=lambda _: scorer,
    )
    with TestClient(app) as client:
        payload = {
            "model": MODEL_ALIAS,
            "query": "question",
            "documents": ["text"],
            "top_n": 1,
            **change,
        }
        assert (
            client.post(
                "/v1/rerank",
                json=payload,
                headers={"Authorization": "Bearer local-test-key"},
            ).status_code
            == 422
        )
    assert scorer.calls == 0


def test_mismatched_model_and_invalid_scores_fail_closed():
    scorer = FakeScorer(None)
    app = create_app(
        LocalRerankerConfig(Path("unused"), "local-test-key"),
        scorer_factory=lambda _: scorer,
    )
    with TestClient(app) as client:
        payload = {
            "model": "wrong-revision",
            "query": "question",
            "documents": ["text"],
            "top_n": 1,
        }
        assert (
            client.post(
                "/v1/rerank",
                json=payload,
                headers={"Authorization": "Bearer local-test-key"},
            ).status_code
            == 400
        )
        scorer.score = lambda *_: [float("nan")]
        payload["model"] = MODEL_ALIAS
        response = client.post(
            "/v1/rerank",
            json=payload,
            headers={"Authorization": "Bearer local-test-key"},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "RERANK_INVALID_SCORES"
