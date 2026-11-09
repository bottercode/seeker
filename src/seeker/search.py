"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, then cross-encoder reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Literal

import lancedb
import numpy as np

from seeker import models
from seeker.config import get_settings
from seeker.quality import host_quality

SearchType = Literal["auto", "neural", "keyword", "fast"]
RESULT_COLUMNS = ["id", "url", "domain", "title", "text", "published", "quality"]


@dataclass
class Filters:
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    include_text: list[str] = field(default_factory=list)
    exclude_text: list[str] = field(default_factory=list)
    start_published_date: str | None = None
    end_published_date: str | None = None
    exclude_urls: list[str] = field(default_factory=list)


@dataclass
class Hit:
    id: str
    url: str
    domain: str
    title: str
    text: str
    published: str | None
    score: float
    vector: np.ndarray | None = None


@lru_cache
def table():
    s = get_settings()
    return lancedb.connect(s.db_path).open_table(s.table)


# ---------- filters ----------


def _q(value: str) -> str:
    """Quote a string literal for a LanceDB SQL filter."""
    return "'" + value.replace("'", "''") + "'"


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _domain_clause(domains: list[str]) -> str:
    parts = []
    for d in domains:
        d = d.lower().strip().removeprefix("https://").removeprefix("http://").removeprefix("www.").strip("/")
        if d:
            parts.append(f"(domain = {_q(d)} OR domain LIKE {_q('%.' + _like_escape(d))})")
    return "(" + " OR ".join(parts) + ")" if parts else ""


def build_where(f: Filters) -> str | None:
    clauses = []
    if inc := _domain_clause(f.include_domains):
        clauses.append(inc)
    if exc := _domain_clause(f.exclude_domains):
        clauses.append(f"NOT {exc}")
    for phrase in f.include_text:
        clauses.append(f"text LIKE {_q('%' + _like_escape(phrase) + '%')}")
    for phrase in f.exclude_text:
        clauses.append(f"text NOT LIKE {_q('%' + _like_escape(phrase) + '%')}")
    if f.start_published_date:
        clauses.append(f"published >= {_q(f.start_published_date[:10])}")
    if f.end_published_date:
        clauses.append(f"published <= {_q(f.end_published_date[:10])}")
    if f.exclude_urls:
        clauses.append("url NOT IN (" + ", ".join(_q(u) for u in f.exclude_urls) + ")")
    return " AND ".join(clauses) or None


# ---------- retrievers ----------


def _run(builder, where: str | None, k: int, score_col: str) -> list[dict]:
    cols = [*RESULT_COLUMNS, score_col]  # naming the score column explicitly avoids lance warnings
    if where:
        builder = builder.where(where, prefilter=True)
    return builder.select(cols).limit(k).to_list()


def vector_search(vec: np.ndarray, where: str | None, k: int) -> list[dict]:
    builder = table().search(vec, vector_column_name="vector").distance_type("cosine")
    rows = _run(builder, where, k, "_distance")
    for r in rows:
        r["_sim"] = 1.0 - float(r["_distance"])
    return rows


def keyword_search(query: str, where: str | None, k: int) -> list[dict]:
    try:
        return _run(table().search(query, query_type="fts", fts_columns="text"), where, k, "_score")
    except Exception:  # empty/stopword-only queries can make the FTS parser fail
        return []


def rrf(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, row in enumerate(lst, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (k + rank)
            rows.setdefault(row["id"], row)
    fused = sorted(rows.values(), key=lambda r: scores[r["id"]], reverse=True)
    for r in fused:
        r["_rrf"] = scores[r["id"]]
    return fused


def url_key(url: str) -> str:
    u = url.lower().split("#", 1)[0]
    u = u.removeprefix("https://").removeprefix("http://").removeprefix("www.")
    return u.rstrip("/")


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = url_key(r["url"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _hit(row: dict, score: float) -> Hit:
    return Hit(
        id=row["id"],
        url=row["url"],
        domain=row["domain"],
        title=row["title"],
        text=row["text"],
        published=row["published"],
        score=score,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------- query rewriting (optional) ----------


def hyde_passage(query: str) -> str | None:
    """Ask a local LLM for a hypothetical answer passage; embedding it often beats the bare query."""
    s = get_settings()
    if not s.hyde_model:
        return None
    import httpx

    prompt = (
        "Write a short, factual web page excerpt (3-4 sentences) that would be a perfect "
        f"search result for this query. No preamble.\n\nQuery: {query}"
    )
    try:
        r = httpx.post(
            f"{s.ollama_url}/api/generate",
            json={"model": s.hyde_model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or None
    except httpx.HTTPError:
        return None


# ---------- public API ----------


def _passes(row: dict, f: Filters) -> bool:
    """Spam + text filters, applied in Python to retrieved candidates (case-insensitive)."""
    s = get_settings()
    if s.spam_filter and host_quality(row["domain"])[1]:
        return False
    if f.include_text or f.exclude_text:
        text = row["text"].lower()
        if any(p.lower() not in text for p in f.include_text):
            return False
        if any(p.lower() in text for p in f.exclude_text):
            return False
    return True


def _candidates(query: str, qvec, type: SearchType, where: str | None, k: int) -> list[dict]:
    """Ranked candidates, each with a comparable score in ``_s``."""
    s = get_settings()
    if type == "keyword":
        rows = keyword_search(query, where, k)
        for r in rows:
            r["_s"] = float(r["_score"])
        return dedupe(rows)
    if type == "neural":
        rows = vector_search(qvec, where, k)
        for r in rows:
            r["_s"] = r["_sim"]
        return dedupe(rows)
    vector_lists = [vector_search(qvec, where, k)]
    if type == "auto" and (passage := hyde_passage(query)):
        vector_lists.append(vector_search(models.embed_docs([passage])[0], where, k))
    fused = dedupe(rrf([*vector_lists, keyword_search(query, where, k)], k=s.rrf_k))
    # Host-quality prior: a small nudge toward well-known / institutional sites.
    for r in fused:
        r["_s"] = r["_rrf"] + s.prior_weight * host_quality(r["domain"])[0]
    return sorted(fused, key=lambda r: r["_s"], reverse=True)


def search(
    query: str,
    num_results: int = 10,
    type: SearchType = "auto",
    filters: Filters | None = None,
) -> list[Hit]:
    s = get_settings()
    f = filters or Filters()
    k = max(s.candidate_k, num_results * 3)
    qvec = None if type == "keyword" else models.embed_query(query)

    # Fast path: SQL-prefilter only the cheap indexed fields; check text filters on candidates.
    cheap = replace(f, include_text=[], exclude_text=[])
    rows = [r for r in _candidates(query, qvec, type, build_where(cheap), k) if _passes(r, f)]
    if (f.include_text or f.exclude_text) and len(rows) < num_results:
        # Too few candidates survived: fall back to a full-table text scan (slower, complete).
        rows = [r for r in _candidates(query, qvec, type, build_where(f), k) if _passes(r, f)]

    if type in ("keyword", "neural", "fast") or s.rerank == "off":
        return [_hit(r, r["_s"]) for r in rows[:num_results]]

    head = rows[: max(s.rerank_depth, num_results)]
    scores = models.rerank_scores(query, [f"{r['title']}\n{r['text'][: s.rerank_chars]}" for r in head])
    if s.rerank == "pure":
        ranked = sorted(zip(head, scores), key=lambda p: p[1], reverse=True)
        return [_hit(r, _sigmoid(sc)) for r, sc in ranked[:num_results]]
    # blend: RRF over (hybrid order, reranker order) so the reranker can't stray far from retrieval.
    by_rerank = sorted(range(len(head)), key=lambda i: -scores[i])
    rerank_pos = {i: pos for pos, i in enumerate(by_rerank)}
    blended = {i: 1 / (s.rrf_k + i + 1) + 1 / (s.rrf_k + rerank_pos[i] + 1) for i in range(len(head))}
    order = sorted(blended, key=lambda i: -blended[i])
    return [_hit(head[i], blended[i]) for i in order[:num_results]]


def lookup(ids: list[str] | None = None, urls: list[str] | None = None) -> dict[str, Hit]:
    """Fetch stored pages by id or URL. Keys are the id/url exactly as requested."""
    out: dict[str, Hit] = {}
    for key, col in [(i, "id") for i in ids or []] + [(u, "url") for u in urls or []]:
        candidates = [key]
        if col == "url":
            bare = url_key(key)
            candidates = list(
                {key, f"http://{bare}", f"https://{bare}", f"http://www.{bare}", f"https://www.{bare}",
                 f"http://{bare}/", f"https://{bare}/", f"https://www.{bare}/", f"http://www.{bare}/"}
            )
        where = f"{col} IN (" + ", ".join(_q(c) for c in candidates) + ")"
        rows = table().search().where(where).select(RESULT_COLUMNS + ["vector"]).limit(1).to_list()
        if rows:
            hit = _hit(rows[0], 1.0)
            hit.vector = np.asarray(rows[0]["vector"], dtype=np.float32)
            out[key] = hit
    return out


def find_similar(
    url: str,
    num_results: int = 10,
    exclude_source_domain: bool = False,
    filters: Filters | None = None,
    fetch=None,
) -> list[Hit]:
    """Nearest pages to a URL: its stored vector if indexed, else embed the live page text."""
    from seeker.ingest import domain_of

    filters = filters or Filters()
    stored = lookup(urls=[url]).get(url)
    if stored is not None:
        vec = stored.vector
        filters.exclude_urls = [*filters.exclude_urls, stored.url]
    else:
        page = (fetch or _live_fetch)(url)
        if not page:
            return []
        vec = models.embed_docs([page["text"][:4000]])[0]
    if exclude_source_domain:
        filters.exclude_domains = [*filters.exclude_domains, domain_of(url)]
    k = max(get_settings().candidate_k, num_results * 3)
    rows = [
        r
        for r in dedupe(vector_search(vec, build_where(filters), k))
        if url_key(r["url"]) != url_key(url) and _passes(r, Filters())
    ]
    return [_hit(r, r["_sim"]) for r in rows[:num_results]]


def _live_fetch(url: str):
    from seeker.contents import fetch_live

    return fetch_live(url)
