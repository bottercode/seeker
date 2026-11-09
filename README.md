# seeker

Open-source semantic web search for AI agents, with an **Exa-compatible API** and an **MCP server**. Runs on a laptop, costs $0.

Most open-source "search for agents" projects wrap SearXNG (i.e. scrape Google/Bing). seeker has **its own index**: pages from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (quality-filtered Common Crawl), whose precomputed `bge-small-en-v1.5` embeddings we reuse. So there's no crawling and no embedding cost.

## How it works

```
query ─┬─ BM25 (LanceDB FTS) ───────┐
       └─ bge-small vector search ──┴─ Reciprocal Rank Fusion ─ cross-encoder rerank ─ results
                                                                (+ highlights, live fetch)
```

| Search `type` | What it does |
|---|---|
| `keyword` | BM25 only |
| `neural` | dense vectors only |
| `fast` | hybrid (RRF), no reranking |
| `auto` (default) | hybrid; plus cross-encoder rerank if `SEEKER_RERANK=blend\|pure` |

## Quickstart

```bash
git clone https://github.com/bottercode/seeker && cd seeker
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run hf auth login       # free HF account, avoids anonymous rate limits
uv run seeker ingest --n 100000     # ~12 min, ~1.5 GB on disk
uv run seeker search "how do vaccines train the immune system"
uv run seeker serve        # http://127.0.0.1:8000/docs
```

### HTTP API (Exa-shaped)

```bash
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "beginner-friendly explanation of how attention works in transformers",
  "numResults": 5,
  "contents": {"text": {"maxCharacters": 500}, "highlights": true}
}'
```

Endpoints: `POST /search`, `POST /findSimilar`, `POST /contents`, `GET /health`.
Filters: `includeDomains`, `excludeDomains`, `includeText`, `excludeText`, `startPublishedDate`, `endPublishedDate`.

The official Exa SDK works by changing the base URL:

```python
from exa_py import Exa
exa = Exa(api_key="unused", base_url="http://127.0.0.1:8000")
exa.search("how do solar panels work", num_results=5)
```

### MCP (Claude Code, Claude Desktop, …)

```bash
claude mcp add seeker -- uv --directory /path/to/seeker run seeker mcp
```

Tools: `web_search`, `find_similar`, `get_contents`.

## Configuration

Environment variables (`SEEKER_*`) or `.env`. See `src/seeker/config.py`:

| Variable | Default | |
|---|---|---|
| `SEEKER_RERANK` | `off` | `blend` or `pure` to enable reranking in `auto` |
| `SEEKER_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | `BAAI/bge-reranker-base` is slower |
| `SEEKER_RERANK_DEPTH` | `30` | candidates reranked |
| `SEEKER_HYDE_MODEL` | *(off)* | e.g. `llama3.2:3b` via Ollama for query rewriting |
| `SEEKER_DEVICE` | `auto` | `cpu`, `mps`, `cuda` |

## Evaluation

`eval/run_eval.py` runs a known-item retrieval test: a local LLM writes a natural search query for each of N random indexed pages (without copying phrases), then we measure where that page ranks.

```bash
ollama pull llama3.2:3b
uv run python eval/run_eval.py --n 150
```

Results: see [Benchmarks](#benchmarks).

## Benchmarks

Known-item eval, 150 LLM-written queries, 99,973 pages, MacBook Air M2 (8 GB), warm models:

| `type` | Recall@1 | Recall@10 | MRR@10 | p50 latency |
|---|---|---|---|---|
| `keyword` (BM25) | 0.653 | 0.867 | 0.732 | 9 ms |
| `neural` | 0.760 | **0.933** | **0.824** | 39 ms |
| `fast` / `auto` (hybrid) | **0.767** | 0.927 | **0.824** | 45 ms |

Reranker A/B on the same queries (`auto` with `SEEKER_RERANK`):

| Reranker | MRR@10 | p50 |
|---|---|---|
| none (hybrid) | 0.820 | ~45 ms |
| ms-marco-MiniLM-L6 / 256 tok, blend | 0.804 | 230 ms |
| ms-marco-MiniLM-L6 / 512 tok, blend | 0.820 | 500 ms |
| bge-reranker-base / 512 tok, pure | 0.806 | 3300 ms |

Rerankers didn't pay off here. Note that known-item eval only credits the one source page, so it under-rewards rerankers that surface *other* relevant pages. A graded-relevance eval is on the roadmap.

## Limitations

- The index is a snapshot of educational/informational pages, so it's weak on news and very recent events.
- FineWeb has no `<title>` or publish date. Titles are derived heuristically, and `publishedDate` is the Common Crawl crawl date.
- `/answer` and Exa's `category` filter are not implemented yet.
- No spam/host-quality signal yet, so some SEO-spam subdomains show up. Common Crawl host ranks are the planned fix.

## Data & license

Code: MIT. Index data: FineWeb-Edu, [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/), also subject to the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).
