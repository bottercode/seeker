"""MCP server exposing the local index to agents (Claude Code, Claude Desktop, etc.) over stdio."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from seeker import contents as contents_mod
from seeker import search as engine

server = MCPServer(
    name="seeker",
    log_level="WARNING",
    instructions=(
        "Semantic web search over a local open index of high-quality web pages. "
        "Describe the page you want in natural language for best results. "
        "The index is a static snapshot, so it is weak on very recent events."
    ),
)


def _fmt(hits: list[engine.Hit], query: str | None, max_chars: int) -> list[dict]:
    out = []
    for h in hits:
        item = {"title": h.title, "url": h.url, "published": h.published, "score": round(h.score, 4)}
        if query:
            hl, _ = contents_mod.highlights(query, h.text, num_sentences=3, per_url=2)
            item["highlights"] = hl
        item["text"] = h.text[:max_chars]
        out.append(item)
    return out


@server.tool()
def web_search(
    query: str,
    num_results: int = 8,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    max_characters: int = 1500,
) -> list[dict]:
    """Search the web index. Returns title, url, crawl date, relevant highlights and the start of each page."""
    filters = engine.Filters(
        include_domains=include_domains or [],
        exclude_domains=exclude_domains or [],
        start_published_date=start_published_date,
    )
    hits = engine.search(query, min(num_results, 25), "auto", filters)
    return _fmt(hits, query, max_characters)


@server.tool()
def find_similar(url: str, num_results: int = 8, exclude_source_domain: bool = True) -> list[dict]:
    """Find pages similar in meaning to the given URL."""
    hits = engine.find_similar(url, min(num_results, 25), exclude_source_domain)
    return _fmt(hits, None, 800)


@server.tool()
def get_contents(url: str, max_characters: int = 8000) -> dict:
    """Get the full text of a page: from the index if present, otherwise fetched live."""
    hit = engine.lookup(urls=[url]).get(url)
    if hit:
        return {"url": hit.url, "title": hit.title, "text": hit.text[:max_characters], "source": "index"}
    page = contents_mod.fetch_live(url)
    if not page:
        return {"url": url, "error": "could not fetch page"}
    return {**page, "text": page["text"][:max_characters], "source": "live"}


def run() -> None:
    server.run("stdio")
