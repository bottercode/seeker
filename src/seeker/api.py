"""HTTP API mirroring Exa's request/response shapes, so Exa SDKs can point at it via base_url."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from seeker import contents as contents_mod
from seeker import search as engine


class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class TextOptions(Camel):
    max_characters: int | None = None


class HighlightOptions(Camel):
    num_sentences: int = 3
    highlights_per_url: int = 1
    query: str | None = None


class ContentsOptions(Camel):
    text: bool | TextOptions | None = None
    highlights: bool | HighlightOptions | None = None


class FilterFields(Camel):
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    include_text: list[str] = Field(default_factory=list)
    exclude_text: list[str] = Field(default_factory=list)
    start_published_date: str | None = None
    end_published_date: str | None = None
    # Our only date is the crawl date, so crawl-date filters map onto the same field.
    start_crawl_date: str | None = None
    end_crawl_date: str | None = None

    def to_filters(self) -> engine.Filters:
        return engine.Filters(
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            include_text=self.include_text,
            exclude_text=self.exclude_text,
            start_published_date=self.start_published_date or self.start_crawl_date,
            end_published_date=self.end_published_date or self.end_crawl_date,
        )


class SearchRequest(FilterFields):
    query: str = Field(min_length=1, max_length=2000)
    num_results: int = Field(default=10, ge=1, le=100)
    type: Literal["auto", "neural", "keyword", "fast"] = "auto"
    contents: ContentsOptions | None = None
    # Exa SDKs may also send contents options at the top level.
    text: bool | TextOptions | None = None
    highlights: bool | HighlightOptions | None = None


class FindSimilarRequest(FilterFields):
    url: str
    num_results: int = Field(default=10, ge=1, le=100)
    exclude_source_domain: bool = False
    contents: ContentsOptions | None = None
    text: bool | TextOptions | None = None
    highlights: bool | HighlightOptions | None = None


class ContentsRequest(Camel):
    ids: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    text: bool | TextOptions | None = None
    highlights: bool | HighlightOptions | None = None
    livecrawl: Literal["never", "fallback", "always", "preferred", "auto"] = "fallback"
    query: str | None = None  # used for highlights when no highlight query is given


class Result(Camel):
    id: str
    url: str
    title: str | None = None
    score: float | None = None
    published_date: str | None = None
    author: str | None = None
    text: str | None = None
    highlights: list[str] | None = None
    highlight_scores: list[float] | None = None


class SearchResponse(Camel):
    request_id: str
    resolved_search_type: str | None = None
    results: list[Result]
    search_time_ms: float | None = None


def _merge_contents(req) -> ContentsOptions:
    c = getattr(req, "contents", None) or ContentsOptions()
    return ContentsOptions(
        text=c.text if c.text is not None else req.text,
        highlights=c.highlights if c.highlights is not None else req.highlights,
    )


def _render(hit: engine.Hit, opts: ContentsOptions, query: str | None, score: bool = True) -> Result:
    res = Result(
        id=hit.id,
        url=hit.url,
        title=hit.title,
        score=round(hit.score, 6) if score else None,
        published_date=hit.published,
    )
    if opts.text:
        max_chars = opts.text.max_characters if isinstance(opts.text, TextOptions) else None
        res.text = hit.text[:max_chars] if max_chars else hit.text
    if opts.highlights:
        h = opts.highlights if isinstance(opts.highlights, HighlightOptions) else HighlightOptions()
        hq = h.query or query
        if hq:
            res.highlights, res.highlight_scores = contents_mod.highlights(
                hq, hit.text, h.num_sentences, h.highlights_per_url
            )
    return res


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and open the table before serving, so the first request isn't ~7 s slower.
    await run_in_threadpool(engine.search, "warm up", 1, "auto")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="seeker",
        version="0.1.0",
        description="Open-source semantic web search, Exa-compatible API.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "pages": engine.table().count_rows()}

    @app.post("/search", response_model=SearchResponse, response_model_exclude_none=True, response_model_by_alias=True)
    async def search(req: SearchRequest):
        t = time.perf_counter()
        opts = _merge_contents(req)

        def work():
            hits = engine.search(req.query, req.num_results, req.type, req.to_filters())
            return [_render(h, opts, req.query) for h in hits]

        results = await run_in_threadpool(work)
        return SearchResponse(
            request_id=uuid.uuid4().hex,
            resolved_search_type=req.type,
            results=results,
            search_time_ms=round((time.perf_counter() - t) * 1000, 1),
        )

    @app.post("/findSimilar", response_model=SearchResponse, response_model_exclude_none=True, response_model_by_alias=True)
    async def find_similar(req: FindSimilarRequest):
        t = time.perf_counter()
        opts = _merge_contents(req)

        def work():
            hits = engine.find_similar(req.url, req.num_results, req.exclude_source_domain, req.to_filters())
            return [_render(h, opts, None) for h in hits]

        results = await run_in_threadpool(work)
        return SearchResponse(
            request_id=uuid.uuid4().hex,
            results=results,
            search_time_ms=round((time.perf_counter() - t) * 1000, 1),
        )

    @app.post("/contents", response_model=SearchResponse, response_model_exclude_none=True, response_model_by_alias=True)
    async def get_contents(req: ContentsRequest):
        opts = ContentsOptions(text=req.text if req.text is not None else True, highlights=req.highlights)

        def work():
            found = engine.lookup(ids=req.ids, urls=req.urls)
            out = []
            for key in [*req.ids, *req.urls]:
                hit = None if req.livecrawl == "always" else found.get(key)
                if hit is None and key in req.urls and req.livecrawl != "never":
                    page = contents_mod.fetch_live(key)
                    if page:
                        hit = engine.Hit(
                            id=page["url"], url=page["url"], domain="", title=page["title"],
                            text=page["text"], published=page["published"], score=1.0,
                        )
                if hit is not None:
                    out.append(_render(hit, opts, req.query, score=False))
            return out

        results = await run_in_threadpool(work)
        return SearchResponse(request_id=uuid.uuid4().hex, results=results)

    # Exa's SDKs also call these paths.
    app.add_api_route("/find_similar", find_similar, methods=["POST"], include_in_schema=False)
    return app


app = create_app()
