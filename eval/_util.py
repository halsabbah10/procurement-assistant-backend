"""Shared helpers for the golden-query eval scripts (run_eval.py,
run_eval_live.py)."""

# Claude's prose sometimes uses typographic dash variants (e.g. a
# non-breaking hyphen, U+2011, in "2014‑2015" instead of a plain ASCII
# hyphen) — cosmetically nicer, but it makes a naive substring match against
# a golden string like "2014-2015" fail even though the answer is correct.
# Verified live: run_eval_live.py's "highest-spending-quarter" case failed
# on exactly this before normalization was added, despite the number being
# right. This does not affect query correctness — fiscal_year values in
# MongoDB are always plain ASCII — it's purely a prose-matching gap.
_DASH_VARIANTS = "‐‑‒–—−"


def normalize_dashes(text: str) -> str:
    for dash in _DASH_VARIANTS:
        text = text.replace(dash, "-")
    return text
