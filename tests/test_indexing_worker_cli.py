from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pytest

from backend.documents.worker import IndexingWorker, IndexingWorkerConfig
import backend.workers.indexing as indexing_cli


ROOT = Path(__file__).resolve().parents[1]


def _cleanup_build(*, status: str = "dead_letter", max_attempts: int = 3):
    return SimpleNamespace(
        job=SimpleNamespace(
            id="cleanup-job-1",
            status=status,
            current_step="object_store",
            attempts=3,
            max_attempts=max_attempts,
            execution_fence=7,
            error_code="STORAGE_UNAVAILABLE",
            next_retry_at=None,
            updated_at=datetime(2026, 7, 16, 3, 4, 5, tzinfo=UTC),
        ),
        document=SimpleNamespace(
            id="document-1",
            tenant_id="tenant-a",
            canonical_name="guide.pdf",
        ),
        version=SimpleNamespace(id="version-1"),
    )


class _OperatorCatalog:
    def __init__(self) -> None:
        self.list_calls: list[dict] = []
        self.requeue_calls: list[dict] = []

    def list_cleanup_jobs(self, **kwargs):
        self.list_calls.append(kwargs)
        return [_cleanup_build()]

    def requeue_cleanup_job(self, **kwargs):
        self.requeue_calls.append(kwargs)
        return _cleanup_build(status="pending", max_attempts=7)


def _install_operator_runtime(monkeypatch, catalog: _OperatorCatalog) -> list[str]:
    startup_calls: list[str] = []
    settings = SimpleNamespace(
        observability=SimpleNamespace(log_level="INFO"),
        validate_startup=lambda: startup_calls.append("validate_startup"),
    )
    monkeypatch.setattr(indexing_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        indexing_cli,
        "init_db",
        lambda: startup_calls.append("init_db"),
    )
    monkeypatch.setattr(indexing_cli, "DocumentCatalog", lambda: catalog)
    return startup_calls


def test_list_cleanup_dead_letter_calls_catalog_and_prints_json(monkeypatch, capsys):
    catalog = _OperatorCatalog()
    startup_calls = _install_operator_runtime(monkeypatch, catalog)

    exit_code = indexing_cli.main(
        [
            "list-cleanup",
            "--status",
            "dead_letter",
            "--tenant-id",
            "tenant-a",
            "--limit",
            "25",
        ]
    )

    assert exit_code == 0
    assert startup_calls == ["validate_startup", "init_db"]
    assert catalog.list_calls == [
        {
            "status": "dead_letter",
            "tenant_id": "tenant-a",
            "limit": 25,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "job_id": "cleanup-job-1",
            "tenant_id": "tenant-a",
            "document_id": "document-1",
            "document_version_id": "version-1",
            "filename": "guide.pdf",
            "status": "dead_letter",
            "current_step": "object_store",
            "attempts": 3,
            "max_attempts": 3,
            "execution_fence": 7,
            "error_code": "STORAGE_UNAVAILABLE",
            "next_retry_at": None,
            "updated_at": "2026-07-16T03:04:05+00:00",
        }
    ]


def test_requeue_cleanup_calls_catalog_with_max_attempts_and_prints_json(
    monkeypatch,
    capsys,
):
    catalog = _OperatorCatalog()
    startup_calls = _install_operator_runtime(monkeypatch, catalog)

    exit_code = indexing_cli.main(
        [
            "requeue-cleanup",
            "--job-id",
            "cleanup-job-1",
            "--max-attempts",
            "7",
        ]
    )

    assert exit_code == 0
    assert startup_calls == ["validate_startup", "init_db"]
    assert catalog.requeue_calls == [
        {
            "job_id": "cleanup-job-1",
            "max_attempts": 7,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "cleanup-job-1"
    assert payload["status"] == "pending"
    assert payload["max_attempts"] == 7


def test_worker_entry_loads_env_before_importing_runtime_modules():
    source_path = ROOT / "backend" / "workers" / "indexing.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    load_env_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load_env"
    )
    runtime_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("backend.")
        and node.module != "backend.env"
    ]

    assert runtime_imports
    assert all(load_env_line < node.lineno for node in runtime_imports)


def _install_pool_runtime(monkeypatch, tmp_path, worker_factory):
    lifecycle = []
    settings = SimpleNamespace(
        observability=SimpleNamespace(log_level="INFO"),
        storage=SimpleNamespace(upload_dir=tmp_path / "uploads"),
        validate_startup=lambda: None,
    )
    monkeypatch.setattr(indexing_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(indexing_cli, "init_db", lambda: None)
    monkeypatch.setattr(indexing_cli.signal, "signal", lambda *_: None)
    monkeypatch.setattr(indexing_cli, "IndexingWorker", worker_factory)
    monkeypatch.setattr(
        indexing_cli,
        "provider_runtime",
        SimpleNamespace(
            start_sync=lambda: lifecycle.append("start"),
            close_sync=lambda: lifecycle.append("close"),
        ),
    )
    return lifecycle


def test_concurrent_claimers_share_one_provider_lifecycle(monkeypatch, tmp_path):
    barrier = Barrier(3)
    lock = Lock()
    thread_ids = set()
    drained = []

    class Worker:
        def run_forever(self, stop_event):
            with lock:
                thread_ids.add(get_ident())
            barrier.wait(timeout=5)
            stop_event.set()
            with lock:
                drained.append(self)

    lifecycle = _install_pool_runtime(monkeypatch, tmp_path, Worker)
    assert indexing_cli.main(["run", "--concurrency", "3"]) == 0
    assert len(thread_ids) == len(drained) == 3
    assert lifecycle == ["start", "close"]


@pytest.mark.parametrize("failure", ["exception", "unexpected_return"])
def test_pool_failure_drains_other_claimers_before_closing_runtime(
    monkeypatch, tmp_path, failure
):
    barrier = Barrier(2)
    workers = []
    drained = Event()

    class Worker:
        def __init__(self):
            self.number = len(workers)
            workers.append(self)

        def run_forever(self, stop_event):
            barrier.wait(timeout=5)
            if self.number == 0:
                if failure == "exception":
                    raise RuntimeError("injected worker failure")
                return
            assert stop_event.wait(timeout=5)
            drained.set()

    lifecycle = _install_pool_runtime(monkeypatch, tmp_path, Worker)
    with pytest.raises(RuntimeError, match="worker"):
        indexing_cli.main(["run", "--concurrency", "2"])
    assert drained.is_set()
    assert lifecycle == ["start", "close"]


@pytest.mark.parametrize("count", ["0", "17"])
def test_pool_rejects_unbounded_concurrency_before_starting_providers(
    monkeypatch, tmp_path, count
):
    lifecycle = _install_pool_runtime(monkeypatch, tmp_path, lambda: None)
    with pytest.raises(SystemExit):
        indexing_cli.main(["run", "--concurrency", count])
    assert lifecycle == []


class _DrainCatalog:
    def __init__(self, *, stop_event: Event, first_claim: str) -> None:
        self.stop_event = stop_event
        self.first_claim = first_claim
        self.claim_calls: list[str] = []
        self.worker_heartbeats: list[dict] = []

    def record_worker_heartbeat(self, **kwargs) -> None:
        self.worker_heartbeats.append(kwargs)

    def _claim(self, kind: str):
        self.claim_calls.append(kind)
        if kind == self.first_claim:
            self.stop_event.set()
        return None

    def claim_index_job(self, **_kwargs):
        return self._claim("index")

    def claim_cleanup_job(self, **_kwargs):
        return self._claim("cleanup")


@pytest.mark.parametrize(
    ("first_claim", "prefer_cleanup"),
    [
        ("index", False),
        ("cleanup", True),
    ],
)
def test_run_once_stops_before_the_next_claim_when_drain_starts(
    first_claim,
    prefer_cleanup,
):
    stop_event = Event()
    catalog = _DrainCatalog(stop_event=stop_event, first_claim=first_claim)
    worker = IndexingWorker(
        catalog=catalog,
        publication=SimpleNamespace(),
        worker_id="index-worker-a",
        config=IndexingWorkerConfig(
            poll_seconds=0.01,
            lease_seconds=30,
            heartbeat_seconds=10,
            retry_base_seconds=1,
            retry_max_seconds=10,
            retry_jitter_ratio=0,
        ),
    )
    worker._prefer_cleanup = prefer_cleanup

    worked = worker.run_once(stop_event)

    assert worked is False
    assert stop_event.is_set()
    assert catalog.claim_calls == [first_claim]
