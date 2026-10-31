"""Explicit live operations; all work crosses existing publication/job services."""

from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path
from uuid import uuid4

from backend.documents.publication import DocumentPublication
from backend.evaluation.crud import news_html, verified_corpus, write_json
from backend.security.uploads import StoredUpload


def ingest_prepared(
    directory: Path,
    *,
    publication: DocumentPublication,
    owner_id: int,
) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    receipt_path = directory / "ingestion.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old["corpus_fingerprint"] != manifest["corpus_fingerprint"]:
            raise ValueError(
                "existing ingestion belongs to a different prepared corpus"
            )
        if old["knowledge_base_name"] != publication.config.knowledge_base_name:
            raise ValueError("existing ingestion belongs to a different Knowledge Base")
    publication.config.upload_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    knowledge_base_id = None
    # Native content/build idempotency makes re-submission safe after interruption.
    articles = iter(verified_corpus(directory))
    while batch := tuple(islice(articles, 100)):
        uploads = []
        for article in batch:
            payload = news_html(article["text"])
            if hashlib.sha256(payload).hexdigest() != article["sha256"]:
                raise ValueError("prepared HTML checksum mismatch")
            object_key = f"{uuid4().hex}.html"
            path = publication.config.upload_dir / object_key
            path.write_bytes(payload)
            uploads.append(
                StoredUpload(
                    original_name=article["name"],
                    object_key=object_key,
                    path=path,
                    extension=".html",
                    media_type="text/html",
                    size_bytes=len(payload),
                    content_sha256=article["sha256"],
                )
            )
        reservations = publication.submit_batch(uploads, owner_id=owner_id)
        knowledge_base_id = reservations[0].document.knowledge_base_id
        count += len(reservations)
        if count % 1000 == 0:
            write_json(
                receipt_path,
                {
                    "corpus_fingerprint": manifest["corpus_fingerprint"],
                    "knowledge_base_name": publication.config.knowledge_base_name,
                    "knowledge_base_id": knowledge_base_id,
                    "submitted_documents": count,
                    "complete": False,
                },
            )
    if count != manifest["document_count"]:
        raise ValueError("prepared corpus document count mismatch")
    receipt = {
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "knowledge_base_name": publication.config.knowledge_base_name,
        "knowledge_base_id": knowledge_base_id,
        "submitted_documents": count,
        "complete": True,
    }
    write_json(receipt_path, receipt)
    return receipt
