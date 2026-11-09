"""Known-item retrieval eval: can the engine find a page from a query a human might type?

1. Sample N indexed pages.
2. A local LLM (Ollama) writes a natural search query for each page, avoiding copied phrases.
3. Run every search type and measure where the source page lands.

Metrics: Recall@1, Recall@10, MRR@10, median latency. Queries are cached so runs are comparable.

    uv run python eval/run_eval.py --n 150 --model llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import httpx

from seeker import search as engine

HERE = Path(__file__).parent
PROMPT = """Here is a web page. Write ONE search query (5-12 words) that a person who has NOT seen this page would type into a search engine, where this page would be a great result.
Describe the information need in your own words. Do NOT copy distinctive phrases or the title from the page. Output only the query.

PAGE:
{text}"""


def gen_query(text: str, model: str, url: str) -> str:
    r = httpx.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": PROMPT.format(text=text[:2500]), "stream": False,
              "options": {"temperature": 0.3, "num_predict": 40}},
        timeout=120,
    )
    r.raise_for_status()
    return clean_query(r.json()["response"])


def clean_query(raw: str) -> str:
    """Small LLMs sometimes add labels or quotes around the query."""
    q = raw.strip().splitlines()[0].strip()
    for prefix in ("Search query:", "Query:", "search query:", "query:"):
        q = q.removeprefix(prefix).strip()
    return q.strip('"').strip()


def load_or_make_queries(n: int, model: str, ollama: str, seed: int) -> list[dict]:
    path = HERE / f"known_item_{n}_seed{seed}.jsonl"
    if path.exists():
        items = [json.loads(line) for line in path.read_text().splitlines()]
        return [{**it, "query": clean_query(it["query"])} for it in items]
    rows = (
        engine.table().search().where("token_count >= 200 AND token_count <= 3000")
        .select(["id", "url", "text"]).limit(20_000).to_list()
    )
    sample = random.Random(seed).sample(rows, n)
    out = []
    for i, row in enumerate(sample, 1):
        q = gen_query(row["text"], model, ollama)
        out.append({"id": row["id"], "url": row["url"], "query": q})
        print(f"  [{i}/{n}] {q}")
    path.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    return out


def evaluate(items: list[dict], types: list[str]) -> dict:
    engine.search("warm up the models", 3, "auto")
    report = {}
    for t in types:
        ranks, lat = [], []
        for it in items:
            s = time.perf_counter()
            hits = engine.search(it["query"], 10, t)
            lat.append((time.perf_counter() - s) * 1000)
            ids = [h.id for h in hits]
            ranks.append(ids.index(it["id"]) + 1 if it["id"] in ids else None)
        n = len(items)
        report[t] = {
            "recall@1": round(sum(r == 1 for r in ranks) / n, 3),
            "recall@10": round(sum(r is not None for r in ranks) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
            "p50_ms": round(statistics.median(lat), 1),
        }
        print(f"{t:8} {report[t]}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--types", default="keyword,neural,fast,auto")
    a = ap.parse_args()
    items = load_or_make_queries(a.n, a.model, a.ollama, a.seed)
    rep = evaluate(items, a.types.split(","))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"known_item_{int(time.time())}.json").write_text(json.dumps({"n": len(items), "pages": engine.table().count_rows(), "results": rep}, indent=2))
