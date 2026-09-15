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
        subsidiary_keywords=[],  # unresearched -- no false-classification risk from
                                  # inheriting axis's names since those now live above,
                                  # but kotak's own subsidiary names (e.g. Kotak
                                  # Securities, Kotak AMC, Kotak Life, Kotak Prime)
                                  # aren't yet enumerated here. Add them when/if that
                                  # topic bucket is validated for this bank.
    ),
    "indusind": BankConfig(
        bank_id="indusind",
        display_name="IndusInd Bank",
        legal_name="IndusInd Bank Limited",
        subsidiary_keywords=[],  # same caveat as kotak above.
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
