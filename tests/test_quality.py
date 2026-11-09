import pytest

from seeker.quality import host_quality


@pytest.mark.parametrize(
    "host",
    [
        "lqassignmentbbda.weblogparkeren.info",
        "dotermpaperouzx.thehealthcopywriter.me",
        "cxessayqvec.rossienginesusa.us",
        "writemyessaynow.net",
        "db0nus869y26v.unknownhost-xyz.net",
    ],
)
def test_spam_hosts(host):
    assert host_quality(host)[1] is True


@pytest.mark.parametrize(
    "host",
    [
        "australianpestspecialists.com.au",
        "caia2023.coloradoadvocacy.org",
        "scholarworks.boisestate.edu",
        "solarsystem.nasa.gov",
        "british-history.ac.uk",
        "my-small-blog.net",
        "socialistvoice.ie",
        "spring2019.veryinteractive.net",
        "njhmvc-stage.reasononeinc.com",
    ],
)
def test_not_spam(host):
    assert host_quality(host)[1] is False


def test_institutional_prior_without_tranco():
    assert host_quality("dept.someuni.edu")[0] == 0.5
    assert host_quality("unknown-site-abc.net")[0] == 0.0
