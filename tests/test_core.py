from seeker.ingest import derive_title, domain_of, dump_to_date, plan_chunks
from seeker.search import Filters, build_where, dedupe, rrf, url_key


def test_dump_to_date():
    assert dump_to_date("CC-MAIN-2019-35") == "2019-08-26"
    assert dump_to_date("CC-MAIN-2013-20") == "2013-05-13"
    assert dump_to_date(None) is None
    assert dump_to_date("garbage") is None


def test_domain_and_title():
    assert domain_of("https://www.Example.com/a/b") == "example.com"
    assert derive_title("Short\nHow photosynthesis works in plants\nbody", "http://x.org/p") == "How photosynthesis works in plants"
    assert derive_title("x" * 500, "http://x.org/when-to-seek-help.html") == "When to seek help"
    # boilerplate lines are skipped in favour of a descriptive slug
    assert derive_title("Read more Health News.\nBy: Zion Lee\nbody text here", "http://a.com/2019/how-to-create-a-vaccine/") == "How to create a vaccine"
    assert derive_title("Article\nok.","http://a.com/article.php?id=1") == "Article"
    assert derive_title("- 10 writers who use stream of consciousness", "http://a.com/x") == "10 writers who use stream of consciousness"
    assert derive_title("12/03/2019 posted\nSolar panels explained for beginners", "http://a.com/p") == "Solar panels explained for beginners"


def test_plan_chunks_covers_n_and_respects_sizes():
    chunks = plan_chunks([1000, 50, 1000], n=1200, chunk=250, seed=1)
    assert sum(c[2] for c in chunks) == 1200
    sizes = [1000, 50, 1000]
    for frag, off, lim in chunks:
        assert 0 <= off and off + lim <= sizes[frag]


def test_rrf_rewards_agreement():
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    b = [{"id": "y"}, {"id": "w"}]
    fused = rrf([a, b], k=60)
    assert fused[0]["id"] == "y"  # ranked in both lists
    assert {r["id"] for r in fused} == {"x", "y", "z", "w"}


def test_url_key_and_dedupe():
    assert url_key("https://www.a.com/p/") == url_key("http://a.com/p#frag")
    rows = [{"url": "http://a.com/p"}, {"url": "https://www.a.com/p/"}, {"url": "http://b.com"}]
    assert [r["url"] for r in dedupe(rows)] == ["http://a.com/p", "http://b.com"]


def test_build_where_escapes_and_combines():
    assert build_where(Filters()) is None
    w = build_where(
        Filters(
            include_domains=["https://www.nasa.gov/"],
            exclude_text=["it's 100%"],
            start_published_date="2020-01-01T00:00:00Z",
        )
    )
    assert "domain = 'nasa.gov'" in w and "domain LIKE '%.nasa.gov'" in w
    assert "text NOT LIKE '%it''s 100\\%%'" in w
    assert "published >= '2020-01-01'" in w
