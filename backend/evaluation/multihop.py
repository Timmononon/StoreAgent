"""Prepare the pinned MultiHop-RAG corpus and a deterministic 100-case sample."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from backend.evaluation.crud import news_html, sha256_file, write_json
from backend.evaluation.rag import (
    RagEvalCase,
    RagEvalDataset,
    RagExpectedBehavior,
    RagGoldDocument,
    RagOutcome,
    dataset_fingerprint,
)

MULTIHOP_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
MULTIHOP_FILES = ("corpus.json", "MultiHopRAG.json")
QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")


def download_multihop_source(destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> tuple[str, str]:
        path = destination / name
        if not path.exists():
            url = f"https://hf-mirror.com/datasets/yixuantt/MultiHopRAG/resolve/{MULTIHOP_REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("MultiHop-RAG source exceeded size limit")
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(data)
            temporary.replace(path)
        return name, sha256_file(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = dict(pool.map(fetch, MULTIHOP_FILES))
    manifest = {
        "source": "yixuantt/MultiHopRAG",
        "revision": MULTIHOP_REVISION,
        "files": files,
    }
    lock = destination / "source.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != manifest:
        raise ValueError("cached MultiHop-RAG source checksum mismatch")
    write_json(lock, manifest)
    return {"revision": MULTIHOP_REVISION, "files": len(files)}


def prepare_multihop(
    source: Path,
    output: Path,
    *,
    count: int = 100,
    dev_count: int = 40,
    seed: int = 20261030,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if count < 4 or not 1 <= dev_count < count:
        raise ValueError("use count >= 4 and 1 <= dev_count < count")
    if (output / "manifest.json").exists():
        raise ValueError("output already contains a prepared benchmark")
    hashes = {name: sha256_file(source / name) for name in MULTIHOP_FILES}
    lock_path = source / "source.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("revision") != MULTIHOP_REVISION or lock.get("files") != hashes:
            raise ValueError("MultiHop-RAG source checksum mismatch")
    articles = json.loads((source / "corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((source / "MultiHopRAG.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    digest = hashlib.sha256()
    corpus_path = output / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as stream:
        for article in articles:
            url = str(article["url"]).strip()
            if not url or url in names or not str(article["body"]).strip():
                raise ValueError(
                    "corpus URLs must be unique and article bodies nonempty"
                )
            name = f"multihop-{hashlib.sha256(url.encode()).hexdigest()}.html"
            names[url] = name
            metadata = "\n".join(
                f"{key}: {article.get(key, '')}"
                for key in (
                    "title",
                    "author",
                    "source",
                    "published_at",
                    "category",
                    "url",
                )
            )
            text = metadata + "\n\n" + article["body"]
            payload = news_html(text)
            digest.update(name.encode() + b"\0" + payload + b"\0")
            stream.write(
                json.dumps(
                    {
                        "name": name,
                        "text": text,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    pools: dict[str, list[RagEvalCase]] = {kind: [] for kind in QUESTION_TYPES}
    for row in questions:
        kind = row["question_type"]
        urls = sorted({item["url"].strip() for item in row["evidence_list"]})
        if any(url not in names for url in urls):
            raise ValueError("question evidence URL missing from corpus")
        if (kind == "null_query" and urls) or (kind != "null_query" and len(urls) < 2):
            raise ValueError("question has inconsistent evidence")
        identity = hashlib.sha256(row["query"].encode()).hexdigest()[:24]
        pools[kind].append(
            RagEvalCase(
                id=f"multihop:{kind}:{identity}",
                question=row["query"],
                reference_answer=row["answer"],
                tags=("multihop_rag", kind, f"hop_{len(urls)}"),
                expected=RagExpectedBehavior(
                    outcome=RagOutcome.NO_KNOWLEDGE
                    if kind == "null_query"
                    else RagOutcome.ANSWERABLE,
                    acceptable_abstention=kind == "null_query",
                ),
                gold_documents=tuple(
                    RagGoldDocument(canonical_name=names[url]) for url in urls
                ),
            )
        )
    rng = random.Random(seed)
    selected = []
    for index, kind in enumerate(QUESTION_TYPES):
        quota = count // 4 + int(index < count % 4)
        selected.extend(
            rng.sample(sorted(pools[kind], key=lambda case: case.id), quota)
        )
    dev_ids: set[str] = set()
    for index, kind in enumerate(QUESTION_TYPES):
        quota = dev_count // 4 + int(index < dev_count % 4)
        pool = sorted(
            (case for case in selected if kind in case.tags), key=lambda case: case.id
        )
        dev_ids.update(case.id for case in rng.sample(pool, quota))
    cases = tuple(
        sorted(
            (
                case.model_copy(
                    update={
                        "tags": (
                            *case.tags,
                            "split_dev" if case.id in dev_ids else "split_test",
                        )
                    }
                )
                for case in selected
            ),
            key=lambda case: case.id,
        )
    )
    dataset = RagEvalDataset(
        name=f"multihop_rag_{count}_seed_{seed}_stratified_v2", cases=cases
    )
    splits = {
        split: sorted(case.id for case in cases if f"split_{split}" in case.tags)
        for split in ("dev", "test")
    }
    write_json(output / "dataset.json", dataset.model_dump(mode="json"))
    for split, ids in splits.items():
        part = RagEvalDataset(
            name=f"{dataset.name}_{split}",
            cases=tuple(case for case in cases if case.id in ids),
        )
        write_json(output / f"dataset_{split}.json", part.model_dump(mode="json"))
    manifest = {
        "schema_version": 1,
        "source": "yixuantt/MultiHopRAG",
        "revision": MULTIHOP_REVISION,
        "license": "ODC-BY",
        "seed": seed,
        "case_count": count,
        "splits": splits,
        "document_count": len(names),
        "upstream_case_count": len(questions),
        "selected_by_task": dict(Counter(case.tags[1] for case in cases)),
        "sampling": "balanced-question-types; stratified-case-split; shared-corpus; v2",
        "source_files": hashes,
        "corpus_fingerprint": digest.hexdigest(),
        "corpus_file_sha256": sha256_file(corpus_path),
        "dataset_fingerprint": dataset_fingerprint(dataset),
    }
    if expected_manifest is not None and manifest != expected_manifest:
        raise ValueError("preparation differs from the committed benchmark manifest")
    write_json(output / "manifest.json", manifest)
    return manifest
