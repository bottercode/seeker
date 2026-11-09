"""Build the keyword (BM25) and vector indexes on the local table."""

from __future__ import annotations

import lancedb

from seeker.config import Settings

# Below this, exact (flat) vector search is fast enough and has perfect recall.
ANN_MIN_ROWS = 250_000


def build_indexes(settings: Settings, log=print) -> None:
    db = lancedb.connect(settings.db_path)
    table = db.open_table(settings.table)
    rows = table.count_rows()
    log(f"indexing {rows:,} rows")

    table.create_fts_index("text", replace=True, stem=True, remove_stop_words=True)
    log("  FTS (BM25) index on text: done")

    table.create_scalar_index("url", replace=True)
    table.create_scalar_index("domain", replace=True)
    log("  scalar indexes on url, domain: done")

    if rows >= ANN_MIN_ROWS:
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_HNSW_SQ",
            replace=True,
        )
        log("  vector index (IVF_HNSW_SQ): done")
    else:
        log(f"  vector index skipped (< {ANN_MIN_ROWS:,} rows, flat search is exact and fast)")

    table.optimize()
