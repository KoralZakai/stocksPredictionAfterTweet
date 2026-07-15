"""Entity-mention disambiguation (sector_mapping/entities.py ENTITY_TRIGGERS).

Regression: bare "intel" (the intelligence-community word) must NOT link INTC —
it manufactured a fake INTC event from "US intel secretly flagged..." posts.
"""

import pytest

from sector_mapping.entities import is_direct_mention

INTEL_NOT_THE_COMPANY = [
    "Obama Intel Chief James Clapper told NSA head...",
    "US intel secretly flagged major 2020 election vulnerabilities",
    "American intel community says otherwise",
    "the intel was totally FAKE",
    "Intel briefing at the White House today",
    "Intel officials leaked the report to the Failing New York Times",
]
INTEL_THE_COMPANY = [
    "Intel is building fabs in Arizona",
    "INTEL got a great deal from my Administration",
    "Intel's CEO is doing a fantastic job",
    "Congratulations to Intel on the new chip plant!",
]


@pytest.mark.parametrize("text", INTEL_NOT_THE_COMPANY)
def test_intel_the_word_does_not_link_intc(text: str) -> None:
    assert not is_direct_mention(text, "INTC")


@pytest.mark.parametrize("text", INTEL_THE_COMPANY)
def test_intel_the_company_links_intc(text: str) -> None:
    assert is_direct_mention(text, "INTC")


GOLDMAN_NOT_THE_BANK = [
    "Weak and pathetic Congressman Dan Goldman just lost, BIG!",
    "Congressman Craig Goldman is an incredible Representative of Texas",
    "Dan Goldman is a disgrace",
]


@pytest.mark.parametrize("text", GOLDMAN_NOT_THE_BANK)
def test_goldman_surname_does_not_link_gs(text: str) -> None:
    # "Goldman" is a politician surname; only the full "Goldman Sachs" links GS.
    assert not is_direct_mention(text, "GS")


def test_goldman_sachs_links_gs() -> None:
    assert is_direct_mention("Goldman Sachs upgraded the stock today", "GS")


def test_djt_signature_is_not_a_company_mention() -> None:
    """Trump signs his posts "DJT". That is a signature, not a reference to Trump
    Media stock — 754 of 762 DJT "mentions" in the 2025-26 corpus were the sign-off.

    This REVERSES an earlier decision to let the signature match on the theory that
    "downstream views must treat signature-only hits as non-company content". That
    filter was never written, so the false positives reached every consumer (they
    were 68% of all corporate mentions). The Trump-exposed-ticker use case the old
    comment cited needs no matcher: every post in this corpus is already by Trump.
    """
    assert not is_direct_mention("THE MOST IMPORTANT WEEKEND. ENJOY!  DJT", "DJT")
    assert not is_direct_mention("RELEASE THE WATER, NEWSOM. DJT", "DJT")
    assert not is_direct_mention("Big things ahead. DJT!", "DJT")
    # ...including the signature followed by a link, which he posts constantly.
    assert not is_direct_mention("This is great news. President DJT https://t.co/x", "DJT")
    # But a real reference to the ticker/company mid-text still counts.
    assert is_direct_mention("DJT stock is up 40% today, tremendous!", "DJT")
    assert is_direct_mention("Trump Media & Technology Group announced earnings", "DJT")
    assert is_direct_mention("BIG WIN today for powerhouse TRUTH Social!", "DJT")


def test_trump_media_press_sense_is_not_the_company() -> None:
    """"Anti Trump Media" is the press corps, not Trump Media & Technology."""
    assert not is_direct_mention("all of the Anti Trump Media that covered me", "DJT")
    assert not is_direct_mention("the Trump media coverage was unfair", "DJT")
