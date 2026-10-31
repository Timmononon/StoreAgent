from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.evaluation.crud import load_prepared_dataset, verified_corpus
from backend.evaluation.multihop import (
    MULTIHOP_FILES,
    MULTIHOP_REVISION,
    QUESTION_TYPES,
    prepare_multihop,
)
from backend.evaluation.rag import RagOutcome
from scripts.crud_rag import build_parser


def make_source(root: Path) -> Path:
    root.mkdir()
    articles = [
        {
            "url": f"https://example.org/{n}",
            "title": f"Article {n}",
            "body": f"Original news text {n}.",
        }
        for n in range(4)
    ]
    questions = [
        {
            "query": f"Question {kind} {n}?",
            "answer": "NEVER_INDEX_REFERENCE_ANSWER",
            "question_type": kind,
            "evidence_list": []
            if kind == "null_query"
            else [
                {"url": item["url"], "fact": "NEVER_INDEX_GOLD_FACT"}
                for item in articles[:2]
            ],
        }
        for kind in QUESTION_TYPES
        for n in range(30)
    ]
    for name, data in zip(MULTIHOP_FILES, (articles, questions), strict=True):
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    return root


def test_balanced_100_questions_share_full_corpus_with_exact_gold(tmp_path):
    source = make_source(tmp_path / "source")
    first = prepare_multihop(source, tmp_path / "first")
    second = prepare_multihop(source, tmp_path / "second", expected_manifest=first)
    assert first == second
    assert first["case_count"] == 100
    assert first["document_count"] == 4
    assert first["selected_by_task"] == dict.fromkeys(QUESTION_TYPES, 25)
    assert len(first["splits"]["dev"]) == 40
    assert len(first["splits"]["test"]) == 60
    assert set(first["splits"]["dev"]).isdisjoint(first["splits"]["test"])
    dataset = load_prepared_dataset(tmp_path / "first")
    names = {item["name"] for item in verified_corpus(tmp_path / "first")}
    for case in dataset.cases:
        if "null_query" in case.tags:
            assert case.expected.outcome == RagOutcome.NO_KNOWLEDGE
            assert case.expected.acceptable_abstention
            assert not case.gold_documents
        else:
            assert len(case.gold_documents) == 2
            assert {gold.canonical_name for gold in case.gold_documents} <= names
    text = (tmp_path / "first/corpus.jsonl").read_text(encoding="utf-8")
    assert "NEVER_INDEX_REFERENCE_ANSWER" not in text
    assert "NEVER_INDEX_GOLD_FACT" not in text
    for split, expected in (("dev", 10), ("test", 15)):
        part = load_prepared_dataset(tmp_path / "first", split)
        for kind in QUESTION_TYPES:
            assert sum(kind in case.tags for case in part.cases) == expected


def test_missing_gold_url_rejects_preparation(tmp_path):
    source = make_source(tmp_path / "source")
    path = source / "MultiHopRAG.json"
    questions = json.loads(path.read_text(encoding="utf-8"))
    questions[0]["evidence_list"][0]["url"] = "https://example.org/missing"
    path.write_text(json.dumps(questions), encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="evidence URL missing"):
        prepare_multihop(source, output)
    assert not (output / "manifest.json").exists()


def test_source_and_manifest_drift_are_rejected(tmp_path):
    source = make_source(tmp_path / "source")
    with pytest.raises(ValueError, match="committed benchmark manifest"):
        prepare_multihop(source, tmp_path / "mismatch", expected_manifest={})
    assert not (tmp_path / "mismatch/manifest.json").exists()
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in MULTIHOP_FILES
    }
    (source / "source.json").write_text(
        json.dumps({"revision": MULTIHOP_REVISION, "files": hashes}), encoding="utf-8"
    )
    with (source / "corpus.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_multihop(source, tmp_path / "tampered")


def test_multihop_cli_defaults_are_separate_from_crud():
    args = build_parser("multihop").parse_args(["ingest", "--username", "admin"])
    assert args.prepared.parts[-2:] == ("multihop-rag", "prepared-v2")
    assert args.knowledge_base_name == "MultiHop-RAG benchmark"
    args = build_parser("multihop").parse_args(["enqueue", "--username", "admin"])
    assert args.split == "dev"
    assert args.design == "optimization"
    args = build_parser("multihop").parse_args(["prepare"])
    assert (args.count, args.dev_count) == (100, 40)


def test_adjusting_split_preserves_the_same_100_questions_and_corpus(tmp_path):
    source = make_source(tmp_path / "source")
    old = prepare_multihop(source, tmp_path / "old", dev_count=30)
    new = prepare_multihop(source, tmp_path / "new", dev_count=40)
    before = load_prepared_dataset(tmp_path / "old")
    after = load_prepared_dataset(tmp_path / "new")
    assert {case.id for case in before.cases} == {case.id for case in after.cases}
    assert old["corpus_fingerprint"] == new["corpus_fingerprint"]
    assert old["dataset_fingerprint"] != new["dataset_fingerprint"]
