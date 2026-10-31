"""Prepare, ingest and compare CRUD-RAG using StoreAgent's formal services."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.crud import (  # noqa: E402
    CRUD_FILES,
    CRUD_REVISION,
    prepare_crud,
    write_json,
)


def download_source(destination: Path) -> dict:
    def fetch(name: str) -> tuple[str, str]:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/{CRUD_REVISION}/{name}"
        if not path.exists():
            temporary = path.with_name(path.name + ".download")
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as target,
            ):
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 64 * 1024 * 1024:
                        raise ValueError("upstream source file exceeded size limit")
                    target.write(block)
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return name, digest

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(fetch, CRUD_FILES))
    result = {"revision": CRUD_REVISION, "files": hashes}
    lock = destination / "source.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != result:
        raise ValueError(
            "cached source files differ from their pinned download manifest"
        )
    write_json(lock, result)
    return {"revision": CRUD_REVISION, "files": len(hashes)}


def build_parser(benchmark: str = "crud") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Prepare, ingest and compare {benchmark.upper()}-RAG via StoreAgent services."
    )
    base = PROJECT_ROOT / f".artifacts/{benchmark}-rag"
    prepared = base / ("prepared-v2" if benchmark == "multihop" else "prepared")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="explicitly download the pinned public source"
    )
    download.add_argument("--source", type=Path, default=base / "source")
    prepare = commands.add_parser(
        "prepare", help="align evidence and prepare exactly 100 cases by default"
    )
    prepare.add_argument("--source", type=Path, default=base / "source")
    prepare.add_argument("--output", type=Path, default=prepared)
    prepare.add_argument("--count", type=int, default=100)
    prepare.add_argument(
        "--dev-count", type=int, default=40 if benchmark == "multihop" else 30
    )
    prepare.add_argument("--seed", type=int, default=20261030)
    prepare.add_argument("--expected-manifest", type=Path)
    preflight = commands.add_parser(
        "preflight",
        help="check local services and all three model profiles without model calls",
    )
    preflight.set_defaults(command="preflight")
    preflight.add_argument(
        "--output",
        type=Path,
        default=base / "preflight.json",
    )
    ingest = commands.add_parser(
        "ingest", help="submit all prepared articles as persistent Index Jobs"
    )
    ingest.add_argument("--prepared", type=Path, default=prepared)
    ingest.add_argument("--username", required=True)
    ingest.add_argument(
        "--knowledge-base-name",
        default="CRUD-RAG benchmark"
        if benchmark == "crud"
        else "MultiHop-RAG benchmark",
    )
    enqueue = commands.add_parser(
        "enqueue", help="submit the three comparison Jobs after indexing is complete"
    )
    enqueue.add_argument("--prepared", type=Path, default=prepared)
    enqueue.add_argument("--username", required=True)
    enqueue.add_argument(
        "--design",
        choices=("retrieval", "optimization"),
        default="optimization" if benchmark == "multihop" else "retrieval",
    )
    enqueue.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="dev" if benchmark == "multihop" else "all",
    )
    compare = commands.add_parser("compare", help="export persisted comparison results")
    compare.add_argument("--jobs", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def live_preflight() -> dict:
    import socket
    from urllib.parse import urlsplit

    from backend.env import load_env

    load_env()
    from backend.core.settings import get_settings
    from backend.infra.database import schema_revisions
    from backend.model_control import ModelRole, model_control_service
    from backend.indexing.milvus_client import MilvusSettings

    settings = get_settings()
    checks = {
        "reranker_configured": settings.rerank.enabled,
        "model_credential_configured": bool(settings.models.api_key.get_secret_value()),
    }
    for name, url, default_port in (
        ("postgres", settings.storage.database_url.get_secret_value(), 5432),
        ("redis", settings.storage.redis_url.get_secret_value(), 6379),
        ("milvus", MilvusSettings.from_env().uri, 19530),
    ):
        try:
            endpoint = urlsplit(url)
            with socket.create_connection(
                (endpoint.hostname or "127.0.0.1", endpoint.port or default_port),
                timeout=1,
            ):
                checks[f"{name}_reachable"] = True
        except (OSError, ValueError):
            checks[f"{name}_reachable"] = False
    checks["schema_current"] = False
    checks["model_snapshot_ready"] = False
    if checks["postgres_reachable"]:
        try:
            current, expected = schema_revisions()
            checks["schema_current"] = current == expected
            if checks["schema_current"]:
                model_control_service.runtime_snapshot(
                    required_roles=frozenset(ModelRole)
                )
                checks["model_snapshot_ready"] = True
        except Exception:
            # Preflight exposes only readiness booleans, never driver/provider details.
            pass
    return {"ready": all(checks.values()), "checks": checks, "model_calls": 0}


def _live_command(args: argparse.Namespace) -> dict:
    from backend.env import load_env

    load_env()
    from backend.core.settings import get_settings
    from backend.documents.catalog import DocumentCatalog
    from backend.documents.retrieval import DocumentRetrievalScope
    from backend.evaluation.benchmark import (
        compare_jobs,
        enqueue_comparison,
        render_comparison,
    )
    from backend.evaluation.crud import load_prepared_dataset, verified_corpus
    from backend.evaluation.service import rag_evaluation_service
    from backend.infra.database import init_db

    init_db()
    if args.command == "compare":
        receipt = json.loads(args.jobs.read_text(encoding="utf-8"))
        jobs = tuple(
            rag_evaluation_service.get_job(job_id) for job_id in receipt["job_ids"]
        )
        report = compare_jobs(jobs)
        write_json(args.output.with_suffix(".json"), report)
        args.output.with_suffix(".md").write_text(
            render_comparison(report), encoding="utf-8"
        )
        return {
            "valid": report["valid"],
            "case_count": report["case_count"],
            "report": str(args.output.with_suffix(".json")),
        }
    if args.command == "ingest":
        from dataclasses import replace
        from backend.db.models import User
        from backend.documents.publication import (
            DocumentPublication,
            DocumentPublicationConfig,
        )
        from backend.evaluation.crud_live import ingest_prepared
        from backend.infra.database import SessionLocal

        with SessionLocal() as db:
            user = (
                db.query(User)
                .filter(User.username == args.username, User.role == "admin")
                .first()
            )
            if user is None:
                raise ValueError(
                    "ingestion requires an existing administrator username"
                )
            owner_id = user.id
        config = replace(
            DocumentPublicationConfig.from_runtime(),
            knowledge_base_name=args.knowledge_base_name,
        )
        return ingest_prepared(
            args.prepared,
            publication=DocumentPublication(config=config),
            owner_id=owner_id,
        )

    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    ingestion = json.loads(
        (args.prepared / "ingestion.json").read_text(encoding="utf-8")
    )
    if (
        not ingestion["complete"]
        or ingestion["corpus_fingerprint"] != manifest["corpus_fingerprint"]
    ):
        raise ValueError("all prepared documents must be submitted before comparison")
    dataset = load_prepared_dataset(args.prepared, args.split)
    catalog = DocumentCatalog()
    snapshot = catalog.load_retrieval_snapshot(
        tenant_id=get_settings().app.default_tenant_id,
        knowledge_base_id=ingestion["knowledge_base_id"],
    )
    current = {
        document.canonical_name: document.current_version.content_sha256
        for document in snapshot.documents
        if document.current_version
    }
    expected = {
        article["name"]: article["sha256"] for article in verified_corpus(args.prepared)
    }
    names = set(current)
    gold = {
        item.canonical_name for case in dataset.cases for item in case.gold_documents
    }
    if (
        not gold.issubset(names)
        or current != expected
        or len(names) != manifest["document_count"]
    ):
        raise ValueError(
            "benchmark corpus is incomplete or contains unexpected documents"
        )
    readiness = DocumentRetrievalScope(catalog).resolve(
        tenant_id=get_settings().app.default_tenant_id,
        knowledge_base_id=ingestion["knowledge_base_id"],
    )
    if readiness.current_document_count != manifest["document_count"]:
        raise ValueError("wait for all persistent Index Jobs to finish")
    receipt_path = args.prepared / f"jobs_{args.split}.json"
    if receipt_path.exists():
        raise ValueError(
            "comparison already submitted; inspect its Jobs or choose a new prepared directory"
        )
    record = rag_evaluation_service.create_dataset(
        username=args.username, dataset=dataset
    )
    jobs = enqueue_comparison(
        rag_evaluation_service,
        username=args.username,
        dataset_id=record.id,
        knowledge_base_id=ingestion["knowledge_base_id"],
        corpus_fingerprint=manifest["corpus_fingerprint"],
        design=args.design,
    )
    receipt = {
        "design": args.design,
        "dataset_fingerprint": record.fingerprint,
        "case_count": record.case_count,
        "split": args.split,
        "job_ids": [job.id for job in jobs],
    }
    write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None, *, benchmark: str = "crud") -> int:
    args = build_parser(benchmark).parse_args(argv)
    try:
        if args.command == "download":
            if benchmark == "multihop":
                from backend.evaluation.multihop import download_multihop_source

                result = download_multihop_source(args.source)
            else:
                result = download_source(args.source)
        elif args.command == "prepare":
            expected_path = args.expected_manifest
            if expected_path is None and (args.count, args.dev_count, args.seed) == (
                100,
                40 if benchmark == "multihop" else 30,
                20261030,
            ):
                version = "v2" if benchmark == "multihop" else "v1"
                expected_path = (
                    PROJECT_ROOT
                    / f"evals/rag/{benchmark}_rag_100_{version}.manifest.json"
                )
            expected = (
                json.loads(expected_path.read_text(encoding="utf-8"))
                if expected_path is not None
                else None
            )
            prepare = prepare_crud
            if benchmark == "multihop":
                from backend.evaluation.multihop import prepare_multihop

                prepare = prepare_multihop
            manifest = prepare(
                args.source,
                args.output,
                count=args.count,
                dev_count=args.dev_count,
                seed=args.seed,
                expected_manifest=expected,
            )
            result = {
                key: manifest[key]
                for key in ("case_count", "document_count", "dataset_fingerprint")
            }
        elif args.command == "preflight":
            result = live_preflight()
            write_json(args.output, result)
        else:
            result = _live_command(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("ready", True) else 1
    except Exception as exc:
        # Live driver errors can contain DSNs; expose only their stable type/code.
        detail = (
            str(exc)
            if args.command in {"download", "prepare"} or type(exc) is ValueError
            else getattr(exc, "code", type(exc).__name__)
        )
        print(
            f"{benchmark.upper()}-RAG command failed ({type(exc).__name__}): {detail}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
