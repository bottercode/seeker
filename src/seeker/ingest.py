"""Pull a diverse sample of FineWeb-Edu (with precomputed vectors) into a local LanceDB table."""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from urllib.parse import urlparse

import lance
import lancedb
import pyarrow as pa

from seeker.config import Settings

SOURCE_COLUMNS = ["id", "url", "text", "dump", "token_count", "score", "text_embedding"]

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("url", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("published", pa.string()),  # ISO date, approximated from the crawl dump
        pa.field("quality", pa.float32()),
        pa.field("token_count", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ]
)

_DUMP_RE = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")


def dump_to_date(dump: str | None) -> str | None:
    """CC-MAIN-2019-35 -> Monday of ISO week 35 of 2019 (crawl date, a proxy for publish date)."""
    m = _DUMP_RE.match(dump or "")
    if not m:
        return None
    year, week = int(m.group(1)), max(1, min(52, int(m.group(2))))
    return dt.date.fromisocalendar(year, week, 1).isoformat()


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


_BOILERPLATE = re.compile(
    r"^(by\b|by:|posted|read more|share|home\b|skip|menu|click|published|written|updated|"
    r"date\b|tags?\b|categor|comments?\b|search|log ?in|sign ?(in|up)|subscribe|advertis|"
    r"previous|next\b|back to|print|email|contact|\d{1,4}[/.-]\d|\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))",
    re.IGNORECASE,
)


def _headline(line: str) -> bool:
    words = line.split()
    if not (3 <= len(words) <= 18 and 15 <= len(line) <= 120):
        return False
    if _BOILERPLATE.match(line) or line.endswith((":", ",")):
        return False
    # Long lines ending in a period are body sentences, not headlines.
    return not (line.endswith(".") and len(words) > 10)


def _slug_title(url: str) -> str | None:
    """Readable title from the most descriptive URL path segment, e.g. /how-to-create-a-vaccine/."""
    segments = [s for s in urlparse(url).path.split("/") if s]
    for seg in reversed(segments):
        seg = re.sub(r"\.\w{2,5}$", "", seg)
        words = [w for w in re.split(r"[-_+\s]+|%20", seg) if w]
        alpha = [w for w in words if w.isalpha()]
        if len(alpha) >= 3 and len(alpha) >= 0.6 * len(words):
            return " ".join(alpha).capitalize()
    return None


def derive_title(text: str, url: str) -> str:
    """FineWeb has no <title>: use a headline-like early line, else the URL slug, else the first sentence."""
    lines = [ln.strip().lstrip("-•*–· ").strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    for line in lines[:6]:
        if _headline(line):
            return line
    if slug := _slug_title(url):
        return slug
    first = re.split(r"(?<=[.!?])\s", lines[0], maxsplit=1)[0] if lines else ""
    if first:
        return first if len(first) <= 90 else first[:87].rsplit(" ", 1)[0] + "..."
    return domain_of(url)


def to_record(row: dict) -> dict:
    return {
        "id": row["id"].removeprefix("<urn:uuid:").removesuffix(">"),
        "url": row["url"],
        "domain": domain_of(row["url"]),
        "title": derive_title(row["text"], row["url"]),
        "text": row["text"],
        "published": dump_to_date(row["dump"]),
        "quality": float(row["score"]),
        "token_count": int(row["token_count"]),
        "vector": row["text_embedding"],
    }


def open_or_create(settings: Settings):
    settings.db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(settings.db_path)
    if settings.table in db.table_names():
        return db.open_table(settings.table)
    return db.create_table(settings.table, schema=SCHEMA)


def open_source(settings: Settings):
    """Open the remote dataset, authenticating to Hugging Face if a token is available.

    Anonymous access gets rate-limited quickly; a free HF account (`hf auth login`) avoids that.
    """
    from huggingface_hub import get_token

    token = get_token()
    opts = {"hf_token": token} if token else {}
    return lance.dataset(settings.source_uri, storage_options=opts)


def plan_chunks(fragment_sizes: list[int], n: int, chunk: int, seed: int) -> list[tuple[int, int, int]]:
    """Pick (fragment, offset, limit) slices spread across random fragments (i.e. across years/crawls).

    Small contiguous slices from many fragments keep the sample diverse; fully random rows
    would need one HTTP range request per row and get rate-limited.
    """
    rng = random.Random(seed)
    order = list(range(len(fragment_sizes)))
    rng.shuffle(order)
    chunks, total, rounds = [], 0, 0
    while total < n and rounds < 100:
        for i in order:
            if total >= n:
                break
            take = min(chunk, n - total, fragment_sizes[i])
            if take <= 0:
                continue
            chunks.append((i, rng.randint(0, fragment_sizes[i] - take), take))
            total += take
        rounds += 1
    return chunks


def _read_chunk(fragment, offset: int, limit: int, retries: int = 4) -> list[dict]:
    for attempt in range(retries):
        try:
            return fragment.to_table(columns=SOURCE_COLUMNS, offset=offset, limit=limit).to_pylist()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return []


def ingest(
    settings: Settings, n: int, per_fragment: int = 250, seed: int = 0, workers: int = 4, log=print
) -> int:
    """Sample ~n pages from the source and upsert them (deduped by URL) into the local table."""
    from seeker.quality import ensure_tranco, host_quality

    ensure_tranco(log)
    src = open_source(settings)
    fragments = src.get_fragments()
    log(f"source: {src.count_rows():,} rows in {len(fragments):,} fragments")
    chunks = plan_chunks([f.count_rows() for f in fragments], n, per_fragment, seed)

    table = open_or_create(settings)
    before = table.count_rows()
    # Dedupe by URL in memory (merge_insert per batch rescans the table and gets quadratic).
    seen = set(table.to_lance().to_table(columns=["url"]).column("url").to_pylist()) if before else set()
    fetched, spam, pending = 0, 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_read_chunk, fragments[i], off, lim) for i, off, lim in chunks]
        for done, fut in enumerate(as_completed(futures), 1):
            batch = []
            for r in fut.result():
                if r["url"] in seen or not r["text"] or r["text_embedding"] is None:
                    continue
                seen.add(r["url"])
                rec = to_record(r)
                if host_quality(rec["domain"])[1]:
                    spam += 1
                    continue
                batch.append(rec)
            pending.extend(batch)
            if len(pending) >= 5000 or done == len(futures):
                if pending:
                    table.add(pa.Table.from_pylist(pending, schema=SCHEMA))
                pending = []
            fetched += len(batch)
            if done % 20 == 0 or done == len(futures):
                log(f"  {done}/{len(futures)} chunks, {fetched:,} pages")
    added = table.count_rows() - before
    log(f"skipped {spam:,} spam pages")
    log(f"added {added:,} new pages; table '{settings.table}' now has {table.count_rows():,} rows")
    return added


def retitle(settings: Settings, log=print) -> int:
    """Recompute derived titles in place (after improving derive_title)."""
    db = lancedb.connect(settings.db_path)
    table = db.open_table(settings.table)
    src = table.to_lance().to_table(columns=["id", "url", "text", "title"]).to_pylist()
    changed = [
        {"id": r["id"], "title": t} for r in src if (t := derive_title(r["text"], r["url"])) != r["title"]
    ]
    if changed:
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .execute(pa.Table.from_pylist(changed, schema=pa.schema([("id", pa.string()), ("title", pa.string())])))
        )
    log(f"updated {len(changed):,} of {len(src):,} titles")
    return len(changed)
