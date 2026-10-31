from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from backend.evaluation.crud import (
    CRUD_CASE_FILE,
    CRUD_CORPUS_FILES,
    QA_TASKS,
    news_html,
    load_prepared_dataset,
    prepare_crud,
    verified_corpus,
)
from backend.evaluation.rag import RagEvalDataset
from backend.rag.profiles import RagRetrievalProfile
from backend.runs.request_context import RunRequestContext
from scripts.crud_rag import download_source
from test_rag_retrieval_targets import (
    Embedding,
    RoutedStore,
    TargetStore,
    _doc,
    _snapshot,
    _target,
    _utils,
)


def make_source(root: Path, *, events: int = 140) -> Path:
    data = {task: [] for task in QA_TASKS}
    articles = []
    for event in range(events):
        news = [
            f"第{event:05d}号事件第{n}号原始来源报道。这是足够长的新闻正文，用于验证精确来源映射和分组抽样。"
            for n in range(1, 4)
        ]
        articles.extend(news)
        for index, task in enumerate(QA_TASKS):
            data[task].append(
                {
                    "ID": f"event-{event:05d}",
                    "questions": f"第{event}个事件的{index + 1}个方面是什么？",
                    "answers": f"参考事实{event}",
                    "thoughts": "NEVER_INDEX_THIS_REASONING",
                    **{
                        f"news{n + 1}": text for n, text in enumerate(news[: index + 1])
                    },
                }
            )
    path = root / CRUD_CASE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    raw = ("\n".join(articles) + "\n").encode()
    # Deliberately split inside UTF-8 and inside articles.
    pieces = (raw[:7], raw[7:150], raw[150:])
    for index, name in enumerate(CRUD_CORPUS_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pieces[index] if index < 3 else b"")
    return root


def test_prepare_exactly_100_cases_and_preserve_group_isolation(tmp_path):
    source = make_source(tmp_path / "source")
    first = prepare_crud(source, tmp_path / "first")
    second = prepare_crud(source, tmp_path / "second", expected_manifest=first)
    assert first == second
    assert first["case_count"] == 100
    assert first["document_count"] == 420
    dataset = RagEvalDataset.model_validate_json(
        (tmp_path / "first/dataset.json").read_text(encoding="utf-8")
    )
    assert len({case.id for case in dataset.cases}) == 100
    assert len(first["splits"]["dev"]) == 30
    assert len(first["splits"]["test"]) == 70
    sources = {}
    events = {}
    for split, ids in first["splits"].items():
        cases = [case for case in dataset.cases if case.id in ids]
        sources[split] = {
            gold.canonical_name for case in cases for gold in case.gold_documents
        }
        events[split] = {case.id.split(":")[1] for case in cases}
    assert sources["dev"].isdisjoint(sources["test"])
    assert events["dev"].isdisjoint(events["test"])
    text = (tmp_path / "first/corpus.jsonl").read_text(encoding="utf-8")
    assert "NEVER_INDEX_THIS_REASONING" not in text
    assert "参考事实" not in text
    assert len(list(verified_corpus(tmp_path / "first"))) == 420
    assert load_prepared_dataset(tmp_path / "first") == dataset
    assert len(load_prepared_dataset(tmp_path / "first", "dev").cases) == 30
    assert len(load_prepared_dataset(tmp_path / "first", "test").cases) == 70


def test_split_checksum_rejects_modified_questions_before_submission(tmp_path):
    source = make_source(tmp_path / "source")
    directory = tmp_path / "prepared"
    prepare_crud(source, directory)
    path = directory / "dataset_dev.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"][0]["question"] = "a different question"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="split checksum"):
        load_prepared_dataset(directory, "dev")


def test_prepare_audits_missing_sources_and_detects_corpus_tampering(tmp_path):
    source = make_source(tmp_path / "source")
    data_path = source / CRUD_CASE_FILE
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data[QA_TASKS[0]][0]["news1"] = (
        "这一条足够长但在原始语料中并不存在的参考新闻，会被列入可审查的对齐失败清单。"
    )
    data_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare_crud(source, output)
    assert manifest["rejected_case_count"] == 1
    audit = json.loads((output / "alignment_audit.json").read_text(encoding="utf-8"))
    assert audit["rejected_cases"][0]["reason"] == "missing_or_ambiguous_source"
    with (output / "corpus.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="checksum"):
        list(verified_corpus(output))


def test_html_conversion_treats_source_as_text():
    payload = news_html('<script>alert("injection")</script>')
    assert b"<script>" not in payload
    assert b"&lt;script&gt;" in payload


def test_manifest_mismatch_does_not_publish_a_prepared_benchmark(tmp_path):
    source = make_source(tmp_path / "source", events=30)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="committed benchmark manifest"):
        prepare_crud(source, output, count=12, dev_count=6, expected_manifest={})
    assert not (output / "manifest.json").exists()


def test_ingestion_submits_native_index_jobs_and_resumes_idempotently(tmp_path):
    from backend.documents.catalog import DocumentCatalog
    from backend.documents.publication import (
        DocumentPublication,
        DocumentPublicationConfig,
    )
    from backend.evaluation.crud_live import ingest_prepared
    from test_rag_evaluation_jobs import _environment

    source = make_source(tmp_path / "source", events=70)
    directory = tmp_path / "prepared"
    manifest = prepare_crud(source, directory, count=12, dev_count=6)
    engine, factory, _, _ = _environment()
    catalog = DocumentCatalog(factory)
    config = DocumentPublicationConfig(
        tenant_id="test",
        knowledge_base_name="CRUD-RAG isolated test",
        parser_version="test",
        chunker_version="test",
        embedding_model="test",
        index_version="test",
        vector_collection="test_catalog",
        upload_dir=tmp_path / "uploads",
        max_attempts=1,
    )
    publication = DocumentPublication(catalog=catalog, config=config)
    try:
        first = ingest_prepared(directory, publication=publication, owner_id=1)
        second = ingest_prepared(directory, publication=publication, owner_id=1)
        assert first == second
        assert first["submitted_documents"] == manifest["document_count"] == 210
        documents = catalog.list_documents(
            tenant_id="test", knowledge_base_id=first["knowledge_base_id"], limit=1000
        )
        assert len(documents) == 210
        assert all(document.current_version is None for document in documents)
        assert len(list(config.upload_dir.glob("*.html"))) == 210
    finally:
        engine.dispose()


def test_parallel_download_uses_distinct_temporary_names_for_dotted_parts(
    tmp_path, monkeypatch
):
    import io
    import scripts.crud_rag as cli

    names = tuple(f"data/documents_hallu.txt_part_{n}" for n in range(1, 4))
    monkeypatch.setattr(cli, "CRUD_FILES", names)
    monkeypatch.setattr(
        cli.urllib.request, "urlopen", lambda url, **_: io.BytesIO(url.encode())
    )
    result = download_source(tmp_path)
    assert result["files"] == 3
    for name in names:
        assert (tmp_path / name).read_bytes().endswith(name.encode())
    assert not list(tmp_path.rglob("*.download"))


@pytest.mark.parametrize(
    "mode,rerank", [("dense", False), ("hybrid", False), ("hybrid", True)]
)
def test_three_profiles_use_the_same_retrieval_seam(mode, rerank):
    utils = _utils()
    store = TargetStore(
        dense=[_doc("d", "dense evidence")], hybrid=[_doc("h", "hybrid evidence")]
    )
    utils._milvus_manager = RoutedStore({"catalog": store})
    utils._embedding_service = Embedding()
    called = []

    def rerank_docs(**kwargs):
        called.append(kwargs)
        return kwargs["docs"], {"rerank_enabled": True, "rerank_applied": True}

    utils._rerank_documents = rerank_docs
    result = utils.retrieve_documents(
        "question",
        tenant_id="tenant-a",
        retrieval_snapshot=_snapshot(_target("catalog")),
        profile=RagRetrievalProfile(
            mode=mode, rerank_enabled=rerank, auto_merge_enabled=False
        ),
    )
    assert bool(store.dense_calls) == (mode == "dense")
    assert bool(store.hybrid_calls) == (mode == "hybrid")
    assert bool(called) == rerank
    assert result["meta"]["retrieval_mode"] == mode
    assert result["meta"]["candidate_k"] == 30
    assert result["meta"]["retrieval_top_k"] == 10
    assert not result["meta"].get("rerank_fallback_applied", False)


def test_two_concurrent_contexts_keep_profiles_and_snapshots_isolated():
    barrier = Barrier(2)

    def run(mode):
        context = RunRequestContext.for_sync(
            user_id=mode, thread_id=mode, tenant_id=mode
        )
        context.configure_rag_retrieval_profile(
            RagRetrievalProfile(mode=mode, rerank_enabled=False)
        )
        snapshot = SimpleNamespace(index_id=mode)
        context.get_or_resolve_rag_retrieval_snapshot(lambda: snapshot)
        barrier.wait(timeout=5)
        assert context.rag_retrieval_profile.mode == mode
        with pytest.raises(ValueError, match="rebound"):
            context.configure_rag_retrieval_profile(
                RagRetrievalProfile(mode="hybrid" if mode == "dense" else "dense")
            )
        assert context.get_or_resolve_rag_retrieval_snapshot(lambda: None) is snapshot
        context.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, ("dense", "hybrid")))


@pytest.mark.parametrize(
    "failure", ["rerank_fallback_applied", "retrieval_degraded_code"]
)
def test_successful_rewrite_keeps_prior_retrieval_degradation_visible(failure):
    from backend.evaluation.runtime import RagEvaluationRuntime
    from backend.schemas.rag import normalize_rag_trace

    utils = _utils()
    initial = utils.retrieval_trace_fields(
        {failure: True if failure.endswith("applied") else "HYBRID_UNSUPPORTED"}
    )
    rewritten = utils.retrieval_trace_fields(
        {"rerank_fallback_applied": False}, previous=initial
    )
    trace = normalize_rag_trace(rewritten)
    assert trace["retrieval_degraded"] is True
    assert RagEvaluationRuntime._retrieval_degraded(
        {"rag_trace": {"sub_traces": [trace]}}
    )
