"""
banks.py — the multi-bank registry.

Adding a new bank to the platform means: (1) add an entry here with its
display/legal name and any bank-specific taxonomy keywords, (2) drop its
earnings-call PDFs into earnings_transcript/<bank_id>/, (3) run
`python run.py compile --dataset --bank <bank_id>` then `--graph`. Nothing
else in the pipeline needs to change per-bank -- settings.paths_for(bank_id)
resolves every data path from this registry's bank_id alone.

legal_name feeds dataset_compiler.parse_transcript()'s header-scrub regex
(transcripts repeat the company's full legal name as a running header/footer
that would otherwise pollute the parsed narration/Q&A text).

subsidiary_keywords are ADDITIONAL keywords merged into the shared
"Subsidiaries' Performance" topic bucket (src/graphs/compiler.py TOPICS) at
classification time, on top of the generic subsidiary vocabulary every bank
shares. Axis's own subsidiary names (Axis AMC, Axis Finance, Axis Capital,
Axis Securities, Max Life) used to be hardcoded directly into that shared
dict -- which meant Kotak/IndusInd transcripts were being classified against
AXIS's subsidiary names. Moved here so each bank is graded on its own
subsidiaries only.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BankConfig:
    bank_id: str
    display_name: str
    legal_name: str
    subsidiary_keywords: list[str] = field(default_factory=list)


BANKS: dict[str, BankConfig] = {
    "axis": BankConfig(
        bank_id="axis",
        display_name="Axis Bank",
        legal_name="Axis Bank Limited",
        subsidiary_keywords=[
            "axis amc", "axis finance", "axis capital", "axis securities", "max life",
        ],
    ),
    "kotak": BankConfig(
        bank_id="kotak",
        display_name="Kotak Mahindra Bank",
        legal_name="Kotak Mahindra Bank Limited",
        # Researched 2026-09 (Kotak/IndusInd extraction tuning): unlike Axis,
        # Kotak's narration does a "lending subsidiaries"-style walkthrough
        # most quarters, so getting this list right materially affects which
        # sentences metrics_extractor.py treats as subsidiary- vs bank-level.
        # Names below are ones actually observed appearing in the transcripts
        # (grep'd across all 21 quarters of narration text, not guessed from
        # general knowledge of the group's structure) -- both the short and
        # "Mahindra"-qualified forms are listed separately where both occur,
        # since "kotak mahindra prime" is not a substring of "kotak prime"
        # (or vice versa) for the keyword-matching this list feeds.
        subsidiary_keywords=[
            "kotak securities", "kotak prime", "kotak mahindra prime",
            "kotak amc", "kotak mahindra asset management", "kotak mahindra mutual fund",
            "kotak life", "kotak mahindra capital", "kotak investments",
            "kotak mahindra investments", "kotak general insurance", "kotak alternate assets",
        ],
    ),
    "indusind": BankConfig(
        bank_id="indusind",
        display_name="IndusInd Bank",
        legal_name="IndusInd Bank Limited",
        # Researched 2026-09, same pass as kotak above -- but actually left
        # EMPTY, not unresearched: grepping all 16 quarters of narration for
        # "IndusInd <Capitalized word>" turned up only "IndusInd Easycredit"
        # (2 mentions, ambiguous -- reads like an internal product/vertical
        # name rather than a distinct subsidiary entity being reported on)
        # and no repeated, clearly-separate subsidiary breakdown pattern like
        # Kotak's. Guessing names not actually observed in this bank's own
        # transcripts risks wrongly filtering real bank-level sentences, so
        # this stays empty until a transcript actually narrates one.
        subsidiary_keywords=[],
    ),
}

DEFAULT_BANK = "axis"


def get_bank(bank_id: str) -> BankConfig:
    try:
        return BANKS[bank_id]
    except KeyError:
        raise ValueError(
            f"Unknown bank_id {bank_id!r} -- registered banks: {sorted(BANKS)}"
        ) from None
