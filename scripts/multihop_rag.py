"""MultiHop-RAG commands, sharing the existing persistent benchmark workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.crud_rag import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(benchmark="multihop"))
