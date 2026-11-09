from __future__ import annotations

import json
import time

import typer

from seeker.config import get_settings

app = typer.Typer(help="seeker: open-source semantic web search", no_args_is_help=True)


@app.command()
def ingest(
    n: int = typer.Option(100_000, help="Number of pages to sample from FineWeb-Edu"),
    per_fragment: int = typer.Option(250, help="Pages per source fragment (lower = more diverse, slower)"),
    seed: int = typer.Option(0),
    index: bool = typer.Option(True, help="Build indexes after ingesting"),
):
    """Download a diverse sample of pages (with precomputed embeddings) into the local index."""
    from seeker.ingest import ingest as run_ingest

    s = get_settings()
    t = time.time()
    run_ingest(s, n, per_fragment, seed)
    typer.echo(f"ingest took {time.time() - t:.0f}s")
    if index:
        build_index()


@app.command()
def retitle():
    """Recompute derived page titles for the existing index."""
    from seeker.ingest import retitle as run

    run(get_settings())


@app.command("index")
def build_index():
    """(Re)build BM25 and vector indexes."""
    from seeker.index import build_indexes

    build_indexes(get_settings())


@app.command()
def search(
    query: str,
    n: int = typer.Option(10, "--num", "-n"),
    type: str = typer.Option("auto", help="auto | neural | keyword | fast"),
):
    """Run a search from the terminal."""
    from seeker import search as engine

    engine.search("warmup", 1, type)  # load models so the timing below is honest
    t = time.perf_counter()
    hits = engine.search(query, n, type)  # type: ignore[arg-type]
    ms = (time.perf_counter() - t) * 1000
    for i, h in enumerate(hits, 1):
        typer.echo(f"{i:2}. [{h.score:.3f}] {h.title[:80]}\n    {h.url}")
    typer.echo(f"\n{len(hits)} results in {ms:.0f} ms")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the Exa-compatible HTTP API."""
    import uvicorn

    uvicorn.run("seeker.api:app", host=host, port=port)


@app.command()
def mcp():
    """Start the MCP server on stdio."""
    from seeker.mcp_server import run

    run()


@app.command()
def stats():
    """Show index statistics."""
    from seeker import search as engine

    t = engine.table()
    typer.echo(json.dumps({"rows": t.count_rows(), "indexes": [i.name for i in t.list_indices()]}, indent=2))


def main() -> None:
    app()
