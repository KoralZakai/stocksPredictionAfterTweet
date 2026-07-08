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


def test_djt_signature_still_matches() -> None:
    # \bdjt\b intentionally matches the signature (Trump-exposed ticker, §3);
    # downstream views must treat signature-only hits as non-company content.
    assert is_direct_mention("THE MOST IMPORTANT WEEKEND. ENJOY!  DJT", "DJT")
