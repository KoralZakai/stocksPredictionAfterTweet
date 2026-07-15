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


# --- the "intel" artifact: a ticker rule matching a common English word -------------
# Twin of the DJT-signature bug (754/762 fake corporate mentions). In a political corpus
# "intel" is overwhelmingly INTELLIGENCE: 51 of 55 bare matches in corpus_v3 were the
# spy sense, and they were landing in the CORPORATE cohort of a registered study.
INTELLIGENCE_SENSE = [
    "FBI asked spy agencies to destroy intel on alleged China plot to aid Joe Biden",
    "Comey's media mole needed a 'discount' to deny leaking intel",
    "Obama Intel Chief James Clapper Told NSA Head To Get On Board On Russiagate Intel",
    "Danish intel warned last year about Russian and Chinese military goals",
    "CIA's historic retraction of intel reports exposes political bias",
    "Britain had a meltdown, but U.S. intel kept it secret in America",
    "US intel secretly flagged major 2020 election vulnerabilities",
    "Ex-intel official reported the Hunter Biden laptop letter was a deception operation",
    "Snowden is handing over to Russia a treasure trove of intel. Our politicians are "
    "incapable of dealing!",
    "House Intel Chair: We Cannot Rule Out Sr. Obama Officials",
    "Just concluded a great meeting with my Intel team in the Oval Office",
    "the nation's premier intel-driven law enforcement agency",
    "this intel agent was listening in on Trump's calls",
    # URL slugs have no space for a lookbehind to anchor on — the original guard's blind spot
    "but U.S. intel kept it secret: https://x.com/security/us-intel-has-known-2020-china",
    "DOJ asked to probe: https://x.com/ex-obama-intel-official-referred-hunter-biden",
]

THE_COMPANY = [
    "The CEO of INTEL is highly CONFLICTED and must resign, immediately.",
    "I met with Mr. Lip-Bu Tan, of Intel, along with Secretary of Commerce",
    "the United States of America now fully owns and controls 10% of INTEL",
    "I PAID ZERO FOR INTEL, IT IS WORTH APPROXIMATELY 11 BILLION DOLLARS",
    "a great meeting with the very successful Intel CEO, Lip-Bu Tan. Intel just launched "
    "the first SUB 2 NANOMETER CPU PROCESSOR",
    "Intel Stock continues to rise. I'm very proud of that Company",
    'We all remember "Intel Inside." Stupid Presidents let Taiwan steal our Semiconductor '
    "Factories.",
    # The real 2025-12-08 post, trimmed to the part that carries the context. This rule
    # is precision-first: a bare "AMD, Intel, and other Companies" with NO chip/deal/stock
    # context anywhere in the post is a deliberate miss. That is the price of not tagging
    # every spy-agency tweet as INTC, and it is the right trade in this corpus.
    "NVIDIA's U.S. Customers are already moving forward with their incredible, highly "
    "advanced Blackwell chips, and soon, Rubin, neither of which are part of this deal. "
    "The Department of Commerce is finalizing the details, and the same approach will "
    "apply to AMD, Intel, and other GREAT American Companies.",
]


@pytest.mark.parametrize("text", INTELLIGENCE_SENSE)
def test_intelligence_sense_is_not_the_chipmaker(text: str) -> None:
    """'intel' meaning intelligence must NEVER resolve to INTC.

    These are not hypotheticals — every line is a real corpus post that the previous
    guard tagged INTC, putting spy-agency tweets into a CORPORATE equity cohort.
    """
    assert not is_direct_mention(text, "INTC"), (
        f"intelligence sense mislabelled as the chipmaker: {text[:60]!r}")


@pytest.mark.parametrize("text", THE_COMPANY)
def test_intel_the_company_is_still_found(text: str) -> None:
    """The other half of the fix, and the easy one to lose.

    A guard tightened until nothing matches is not precision, it is deletion. These are
    the real Intel posts — including the whole August-2025 US-stake storyline, which the
    dashboard's old private regex missed entirely because it only looked for context
    AFTER the word.
    """
    assert is_direct_mention(text, "INTC"), (
        f"real Intel-the-company mention lost: {text[:60]!r}")
