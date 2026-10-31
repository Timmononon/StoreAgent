from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace

import pytest

import backend.workers.evaluation as cli


def install_runtime(monkeypatch, worker_factory):
    lifecycle = []
    settings = SimpleNamespace(
        validate_startup=lambda: None, observability=SimpleNamespace(log_level="INFO")
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        cli.model_control_service, "ensure_environment_defaults", lambda: None
    )
    monkeypatch.setattr(cli, "RagEvaluationWorker", worker_factory)
    monkeypatch.setattr(
        cli,
        "provider_runtime",
        SimpleNamespace(
            start_sync=lambda: lifecycle.append("start"),
            close_sync=lambda: lifecycle.append("close"),
        ),
    )
    return lifecycle, settings


def test_three_workers_overlap_and_share_one_runtime_lifecycle(monkeypatch):
    barrier = Barrier(3)
    identities = set()
    settings_seen = []
    lock = Lock()

    class Worker:
        def __init__(self, *, settings):
            settings_seen.append(settings)

        def run_forever(self, stop_event):
            with lock:
                identities.add(get_ident())
            barrier.wait(timeout=5)
            stop_event.set()

    lifecycle, settings = install_runtime(monkeypatch, Worker)
    assert cli.main(["--concurrency", "3"]) == 0
    assert len(identities) == 3
    assert settings_seen == [settings] * 3
    assert lifecycle == ["start", "close"]


@pytest.mark.parametrize("failure", ["exception", "unexpected_return"])
def test_failed_worker_drains_other_workers_before_runtime_closes(monkeypatch, failure):
    barrier = Barrier(2)
    workers = []
    drained = Event()

    class Worker:
        def __init__(self, **_kwargs):
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

    lifecycle, _ = install_runtime(monkeypatch, Worker)
    with pytest.raises(RuntimeError, match="worker"):
        cli.main(["--concurrency", "2"])
    assert drained.is_set()
    assert lifecycle == ["start", "close"]


@pytest.mark.parametrize("count", ["0", "9"])
def test_concurrency_is_bounded_before_runtime_start(monkeypatch, count):
    lifecycle, _ = install_runtime(monkeypatch, lambda **_: None)
    with pytest.raises(SystemExit):
        cli.main(["--concurrency", count])
    assert lifecycle == []
