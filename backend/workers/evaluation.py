from __future__ import annotations

# 独立进程入口必须先把项目 .env 写入 os.environ，再导入运行时模块。
# ruff: noqa: E402

import argparse
import logging
import signal
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

from backend.env import load_env


load_env()

from backend.core.settings import AppSettings, get_settings
from backend.evaluation.worker import RagEvaluationWorker
from backend.infra.database import init_db
from backend.model_control import model_control_service
from backend.providers.runtime import provider_runtime


logger = logging.getLogger(__name__)


def _run_workers(concurrency: int, settings: AppSettings, stop_event: Event) -> None:
    workers = tuple(RagEvaluationWorker(settings=settings) for _ in range(concurrency))
    if concurrency == 1:
        workers[0].run_forever(stop_event)
        return
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="evaluation"
    ) as pool:
        futures = [pool.submit(worker.run_forever, stop_event) for worker in workers]
        try:
            for future in as_completed(futures):
                future.result()
                if not stop_event.is_set():
                    raise RuntimeError(
                        "evaluation worker exited before drain was requested"
                    )
        finally:
            stop_event.set()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run persistent RAG Evaluation workers"
    )
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args(argv)
    settings = get_settings()
    settings.validate_startup()
    logging.basicConfig(
        level=getattr(logging, settings.observability.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    model_control_service.ensure_environment_defaults()
    stop_event = Event()

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("RAG evaluation worker draining", extra={"signal": signum})
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    provider_runtime.start_sync()
    try:
        _run_workers(args.concurrency, settings, stop_event)
        return 0
    finally:
        provider_runtime.close_sync()


if __name__ == "__main__":
    raise SystemExit(main())
