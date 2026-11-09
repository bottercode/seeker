"""Host-level quality signals: a popularity prior (Tranco top-1M) and a spam-host detector.

Tranco (https://tranco-list.eu) is a free research ranking of the top 1M registered domains.
Spam detection targets the commonest junk in Common Crawl: throwaway subdomains on unknown
domains (e.g. ``lqassignmentbbda.weblogparkeren.info``), never well-known or institutional sites.
"""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from functools import lru_cache
from pathlib import Path

from seeker.config import get_settings

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
INSTITUTIONAL = re.compile(r"(^|\.)(gov|edu|mil|ac|gouv|gob|govt)(\.[a-z]{2})?$")
SPAM_WORDS = re.compile(
    r"essay|assignment|termpaper|homework|casino|viagra|(?<!spe)(?<!so)cialis(?![tm])|payday|porn|escort|replica|"
    r"loans?\b|pills|xxx"
)


def tranco_path() -> Path:
    return get_settings().data_dir / "priors" / "top-1m.csv"


def ensure_tranco(log=print) -> Path:
    """Download the Tranco list once (~10 MB) into data/priors/."""
    path = tranco_path()
    if path.exists():
        return path
    import httpx

    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading Tranco top-1M list to {path} ...")
    r = httpx.get(TRANCO_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        path.write_bytes(z.read(z.namelist()[0]))
    return path


@lru_cache
def tranco() -> dict[str, int]:
    path = tranco_path()
    if not path.exists():
        return {}
    with path.open() as f:
        return {domain: int(rank) for rank, domain in csv.reader(f)}


@lru_cache
def _extractor():
    import tldextract

    return tldextract.TLDExtract(suffix_list_urls=())  # bundled public-suffix snapshot, no network


def _gibberish(label: str) -> bool:
    """Random-looking subdomain labels, e.g. ``xkqzvbtrw`` or hash-like ``db0nus869y26v``."""
    if re.search(r"[a-z]\d+[a-z]+\d", label) and sum(c.isdigit() for c in label) >= 3:
        return True
    for token in re.split(r"[-_]", label):
        letters = [c for c in token if c.isalpha()]
        if len(letters) >= 8 and sum(c in "aeiouy" for c in letters) / len(letters) < 0.2:
            return True
    return False


@lru_cache(maxsize=200_000)
def host_quality(host: str) -> tuple[float, bool]:
    """Return (prior in [0, 1], is_spam) for a hostname like ``blog.example.co.uk``."""
    ext = _extractor()(host)
    registered = ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix") else ext.registered_domain
    rank = tranco().get(registered) or tranco().get(host)
    if rank:
        return 1.0 - math.log10(rank) / 6.0, False
    if INSTITUTIONAL.search(ext.suffix or "") or INSTITUTIONAL.search(registered or ""):
        return 0.5, False
    sub_labels = [s for s in (ext.subdomain or "").split(".") if s and s != "www"]
    spam = any(_gibberish(s) or SPAM_WORDS.search(s) for s in sub_labels) or bool(
        SPAM_WORDS.search(ext.domain or "")
    )
    return 0.0, spam
