"""Page contents: stored text, query-relevant highlights, and live fetching for unindexed URLs."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import numpy as np

from seeker import models

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
MAX_WINDOWS = 40
USER_AGENT = "seeker/0.1 (open-source search engine; +https://github.com/bottercode/seeker)"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]


def highlights(query: str, text: str, num_sentences: int = 3, per_url: int = 1) -> tuple[list[str], list[float]]:
    """Return the windows of consecutive sentences most similar to the query."""
    sents = split_sentences(text)
    if not sents:
        return [], []
    step = max(1, num_sentences)
    windows = [" ".join(sents[i : i + num_sentences]) for i in range(0, len(sents), step)][:MAX_WINDOWS]
    q = models.embed_query(query)
    sims = models.embed_docs(windows) @ q
    top = np.argsort(-sims)[:per_url]
    return [windows[i] for i in top], [round(float(sims[i]), 4) for i in top]


def _is_public_host(host: str) -> bool:
    """Block SSRF to localhost / private networks when fetching user-supplied URLs."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch_live(url: str, timeout: float = 10.0) -> dict | None:
    """Download and extract main text with trafilatura. Returns None on any failure."""
    import httpx
    import trafilatura

    r = None
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
            for _ in range(5):  # follow redirects manually so every hop is SSRF-checked
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or not _is_public_host(parsed.hostname):
                    return None
                r = client.get(url)
                if r.is_redirect and "location" in r.headers:
                    url = str(r.url.join(r.headers["location"]))
                    continue
                break
            if r is None or r.is_redirect:
                return None
            r.raise_for_status()
    except httpx.HTTPError:
        return None
    text = trafilatura.extract(r.text, url=url, include_comments=False)
    if not text:
        return None
    meta = trafilatura.extract_metadata(r.text, default_url=url)
    return {
        "url": str(r.url),
        "title": (meta.title if meta and meta.title else url),
        "text": text,
        "published": (meta.date if meta and meta.date else None),
    }
