"""Deterministic preparation of the public CRUD-RAG question-answer benchmark.

This module performs local file I/O only. Downloading and live execution are
explicit commands; importing it never loads a model or opens a network client.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import unicodedata
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.evaluation.rag import (
    RagEvalCase,
    RagEvalDataset,
    RagExpectedBehavior,
    RagGoldDocument,
    RagOutcome,
    dataset_fingerprint,
    load_rag_eval_dataset,
)


CRUD_REVISION = "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"
CRUD_CASE_FILE = "data/crud_split/split_merged.json"
CRUD_CORPUS_FILES = tuple(
    f"data/80000_docs/documents_dup_part_{group}_part_{part}"
    for group in range(1, 16)
    for part in range(1, 4)
) + tuple(f"data/80000_docs/documents_hallu.txt_part_{part}" for part in range(1, 4))
CRUD_FILES = (CRUD_CASE_FILE, *CRUD_CORPUS_FILES)
QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_news(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    if 0 <= text.find("正文:") < 160:
        text = text.split("正文:", 1)[1]
    return "".join(character.lower() for character in text if character.isalnum())


def news_html(text: str) -> bytes:
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"></head>'
        "<body><article><p>" + html.escape(text) + "</p></article></body></html>"
    ).encode("utf-8")


class _Anchors:
    """Aho-Corasick lookup: scan the corpus once for reference passage anchors."""

    def __init__(self, passages: Sequence[str]) -> None:
        self.edges: list[dict[str, int]] = [{}]
        self.failure = [0]
        self.output: list[list[str]] = [[]]
        for passage in sorted(set(passages)):
            node = 0
            for character in passage[:24]:
                if character not in self.edges[node]:
                    self.edges[node][character] = len(self.edges)
                    self.edges.append({})
                    self.failure.append(0)
                    self.output.append([])
                node = self.edges[node][character]
            self.output[node].append(passage)
        queue = deque(self.edges[0].values())
        while queue:
            node = queue.popleft()
            for character, child in self.edges[node].items():
                queue.append(child)
                fallback = self.failure[node]
                while fallback and character not in self.edges[fallback]:
                    fallback = self.failure[fallback]
                self.failure[child] = self.edges[fallback].get(character, 0)
                self.output[child].extend(self.output[self.failure[child]])

    def matching_passages(self, text: str) -> set[str]:
        node = 0
        candidates: set[str] = set()
        for character in text:
            while node and character not in self.edges[node]:
                node = self.failure[node]
            node = self.edges[node].get(character, 0)
            candidates.update(self.output[node])
        return {passage for passage in candidates if passage in text}


def _corpus_lines(source: Path) -> Iterator[str]:
    # Reassemble byte-split files before decoding: a split may bisect UTF-8 or a news item.
    for start in range(0, len(CRUD_CORPUS_FILES), 3):
        raw = b"".join(
            (source / name).read_bytes()
            for name in CRUD_CORPUS_FILES[start : start + 3]
        )
        for text in raw.decode("utf-8-sig").splitlines():
            if text.strip():
                yield text.strip()


def _quota(count: int, capacities: Mapping[str, int]) -> dict[str, int]:
    if sum(capacities.values()) < count:
        raise ValueError("not enough source-aligned cases for the requested split")
    quotas = {
        task: min(capacities[task], count // 3 + int(index < count % 3))
        for index, task in enumerate(QA_TASKS)
    }
    while sum(quotas.values()) < count:
        task = min(
            (task for task in QA_TASKS if quotas[task] < capacities[task]),
            key=lambda task: (quotas[task], QA_TASKS.index(task)),
        )
        quotas[task] += 1
    return quotas


def prepare_crud(
    source: Path,
    output: Path,
    *,
    count: int = 100,
    dev_count: int = 30,
    seed: int = 20261030,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if count < 6 or not 3 <= dev_count <= count - 3:
        raise ValueError("use count >= 6 and 3 <= dev_count <= count - 3")
    if (output / "manifest.json").exists():
        raise ValueError(
            "output already contains a prepared benchmark; choose a new directory"
        )
    data = json.loads((source / CRUD_CASE_FILE).read_text(encoding="utf-8"))
    lock_path = source / "source.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("revision") != CRUD_REVISION or lock.get("files") != {
            name: sha256_file(source / name) for name in CRUD_FILES
        }:
            raise ValueError("downloaded source checksum or revision mismatch")
    records: list[dict[str, Any]] = []
    for task in QA_TASKS:
        for row in data[task]:
            question, answer = (
                str(row["questions"]).strip(),
                str(row["answers"]).strip(),
            )
            passages = [
                normalize_news(str(row[f"news{n}"]))
                for n in range(1, QA_TASKS.index(task) + 2)
            ]
            if not question or not answer or any(len(p) < 24 for p in passages):
                raise ValueError("invalid upstream question, answer, or news passage")
            identity = f"{task}:{row['ID']}:{hashlib.sha256(question.encode()).hexdigest()[:12]}"
            records.append(
                {
                    "id": identity,
                    "event": str(row["ID"]),
                    "task": task,
                    "question": question,
                    "answer": answer,
                    "passages": passages,
                }
            )
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("duplicate upstream case identity")

    anchors = _Anchors([p for row in records for p in row["passages"]])
    matches: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    digest = hashlib.sha256()
    raw_count = 0
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as stream:
        for text in _corpus_lines(source):
            raw_count += 1
            normalized = normalize_news(text)
            if not normalized:
                continue
            identity = hashlib.sha256(normalized.encode()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            name = f"crud-{identity}.html"
            for passage in anchors.matching_passages(normalized):
                matches[passage].add(name)
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

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for row in records:
        if any(len(matches[p]) != 1 for p in row["passages"]):
            rejected.append({"id": row["id"], "reason": "missing_or_ambiguous_source"})
        else:
            row["gold"] = sorted({next(iter(matches[p])) for p in row["passages"]})
            eligible.append(row)
    write_json(
        output / "alignment_audit.json",
        {
            "eligible_cases": len(eligible),
            "eligible_by_task": {
                task: sum(row["task"] == task for row in eligible) for task in QA_TASKS
            },
            "rejected_cases": rejected,
        },
    )

    # Group connected events and shared source articles before selecting any cases.
    parents: dict[str, str] = {}

    def root(value: str) -> str:
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for row in eligible:
        event = "event:" + row["event"]
        for name in row["gold"]:
            first, second = root(event), root(name)
            if first != second:
                parents[max(first, second)] = min(first, second)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {task: [] for task in QA_TASKS} for split in ("dev", "test")
    }
    for row in eligible:
        group = root("event:" + row["event"])
        split = (
            "dev"
            if int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 10 < 3
            else "test"
        )
        pools[split][row["task"]].append(row)
    selected: list[RagEvalCase] = []
    splits: dict[str, list[str]] = {}
    randomizer = random.Random(seed)
    for split, size in (("dev", dev_count), ("test", count - dev_count)):
        ids = []
        for task, requested in _quota(
            size, {task: len(rows) for task, rows in pools[split].items()}
        ).items():
            pool = sorted(pools[split][task], key=lambda row: row["id"])
            if len(pool) < requested:
                raise ValueError(
                    f"not enough aligned {split}/{task} cases: {len(pool)} < {requested}"
                )
            for row in randomizer.sample(pool, requested):
                ids.append(row["id"])
                selected.append(
                    RagEvalCase(
                        id=row["id"],
                        question=row["question"],
                        reference_answer=row["answer"],
                        tags=("crud_rag", task, f"split_{split}"),
                        expected=RagExpectedBehavior(outcome=RagOutcome.ANSWERABLE),
                        gold_documents=tuple(
                            RagGoldDocument(canonical_name=name) for name in row["gold"]
                        ),
                    )
                )
        splits[split] = sorted(ids)
    dataset = RagEvalDataset(
        name=f"crud_rag_{count}_seed_{seed}",
        cases=tuple(sorted(selected, key=lambda case: case.id)),
    )
    write_json(output / "dataset.json", dataset.model_dump(mode="json"))
    for split, ids in splits.items():
        part = RagEvalDataset(
            name=f"{dataset.name}_{split}",
            cases=tuple(case for case in dataset.cases if case.id in ids),
        )
        write_json(output / f"dataset_{split}.json", part.model_dump(mode="json"))
    manifest = {
        "schema_version": 1,
        "source": "IAAR-Shanghai/CRUD_RAG",
        "revision": CRUD_REVISION,
        "seed": seed,
        "case_count": count,
        "splits": splits,
        "raw_document_count": raw_count,
        "eligible_case_count": len(eligible),
        "rejected_case_count": len(rejected),
        "selected_by_task": {
            task: sum(task in case.tags for case in dataset.cases) for task in QA_TASKS
        },
        "document_count": len(seen),
        "corpus_fingerprint": digest.hexdigest(),
        "corpus_file_sha256": sha256_file(corpus_path),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "source_files": {name: sha256_file(source / name) for name in CRUD_FILES},
        "sampling": "source-aligned-only; group-by-event-and-shared-sources; capacity-bounded-stratification; v1",
    }
    if expected_manifest is not None and manifest != expected_manifest:
        raise ValueError("preparation differs from the committed benchmark manifest")
    write_json(output / "manifest.json", manifest)
    return manifest


def verified_corpus(directory: Path) -> Iterator[Mapping[str, str]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "corpus.jsonl"
    if sha256_file(path) != manifest["corpus_file_sha256"]:
        raise ValueError("prepared corpus checksum mismatch")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_prepared_dataset(directory: Path, split: str = "all") -> RagEvalDataset:
    """Validate all/split data against the frozen sample before creating Jobs."""
    if split not in {"all", "dev", "test"}:
        raise ValueError("unknown prepared dataset split")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_rag_eval_dataset(directory / "dataset.json")
    if dataset_fingerprint(dataset) != manifest["dataset_fingerprint"]:
        raise ValueError("prepared dataset checksum mismatch")
    dev_ids, test_ids = (set(manifest["splits"][name]) for name in ("dev", "test"))
    if (
        dev_ids & test_ids
        or dev_ids | test_ids != {case.id for case in dataset.cases}
        or len(dataset.cases) != manifest["case_count"]
    ):
        raise ValueError("prepared split membership mismatch")
    if split == "all":
        return dataset
    expected = RagEvalDataset(
        name=f"{dataset.name}_{split}",
        cases=tuple(
            case for case in dataset.cases if case.id in manifest["splits"][split]
        ),
    )
    part = load_rag_eval_dataset(directory / f"dataset_{split}.json")
    if dataset_fingerprint(part) != dataset_fingerprint(expected):
        raise ValueError("prepared split checksum mismatch")
    return part
