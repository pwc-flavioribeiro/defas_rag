"""
Unit tests for the legal comparison output schema.
Tests run against sample output strings — no Azure credentials required.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "3.Query"))

# ---------------------------------------------------------------------------
# Constants — mirrors query.py
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Legal Unit", "Topic", "Old Version", "New Version",
    "Change Type", "Practical Impact", "Materiality", "Explanation",
]
ALLOWED_CHANGE_TYPES     = {"Added", "Removed", "Unchanged"}
ALLOWED_PRACTICAL_IMPACT = {"None", "Low", "Medium", "High"}
ALLOWED_MATERIALITY      = {"Editorial", "Procedural", "Substantive"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_table_rows(text: str) -> list[dict]:
    """
    Parse a markdown table from the LLM output into a list of row dicts.
    Returns [] if no table is found.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Legal Unit" in line and "Change Type" in line:
            header_idx = i
            break
    if header_idx is None:
        return []

    headers = [h.strip() for h in lines[header_idx].strip("|").split("|")]
    rows = []
    for line in lines[header_idx + 2:]:   # skip separator line
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def has_executive_summary(text: str) -> bool:
    """Check that output starts with prose before the table."""
    table_start = text.find("| Legal Unit |")
    if table_start == -1:
        return False
    preamble = text[:table_start].strip()
    return len(preamble) > 50   # at least some meaningful prose


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_OUTPUT = """\
## Executive Summary

The 2019 version of the Vibration Directive introduces tighter exposure limits and new \
obligations for employers regarding health surveillance. Several provisions were restructured \
from Annex I into the main body of the directive. Editorial changes are limited and do not \
affect legal substance.

| Legal Unit | Topic | Old Version | New Version | Change Type | Practical Impact | Materiality | Explanation |
|---|---|---|---|---|---|---|---|
| Article 3 | Exposure limit values | Daily ELV: 2.5 m/s² | Daily ELV: 1.15 m/s² | Removed | High | Substantive | 2008 version set ELV at 2.5 m/s². Evidence: Art.3(1) old. |
| Article 3 | Exposure limit values | — | Daily ELV: 1.15 m/s² | Added | High | Substantive | 2019 version reduces ELV. Evidence: Art.3(1) new. |
| Article 5 | Health surveillance | Employer shall provide | Employer shall ensure | Unchanged | None | Editorial | Wording differs slightly; legal obligation identical. |
| Annex I | Measurement methods | Referenced ISO 5349 | — | Removed | Medium | Procedural | Annex I removed from 2019 version; standard now referenced in Article 6. |
"""

BAD_OUTPUT_NO_TABLE = """\
The directive changed significantly between 2008 and 2019. Article 3 now contains lower
exposure limits. Annex I was removed. Employers face new obligations.
"""

BAD_OUTPUT_WRONG_CHANGE_TYPE = """\
## Executive Summary
Minor changes only.

| Legal Unit | Topic | Old Version | New Version | Change Type | Practical Impact | Materiality | Explanation |
|---|---|---|---|---|---|---|---|
| Article 3 | ELV | 2.5 m/s² | 1.15 m/s² | Modified | High | Substantive | ELV reduced. |
"""

BAD_OUTPUT_WRONG_IMPACT = """\
## Executive Summary
Minor changes.

| Legal Unit | Topic | Old Version | New Version | Change Type | Practical Impact | Materiality | Explanation |
|---|---|---|---|---|---|---|---|
| Article 3 | ELV | 2.5 m/s² | — | Removed | Critical | Substantive | Old ELV removed. |
"""


# ---------------------------------------------------------------------------
# Tests — output structure
# ---------------------------------------------------------------------------

def test_table_present_in_good_output():
    rows = parse_table_rows(GOOD_OUTPUT)
    assert len(rows) > 0, "Expected at least one data row in the comparison table."


def test_table_has_all_required_columns():
    rows = parse_table_rows(GOOD_OUTPUT)
    for col in REQUIRED_COLUMNS:
        assert col in rows[0], f"Missing required column: '{col}'"


def test_executive_summary_present():
    assert has_executive_summary(GOOD_OUTPUT), "Output must start with an executive summary before the table."


def test_no_table_detected_in_prose_only_output():
    rows = parse_table_rows(BAD_OUTPUT_NO_TABLE)
    assert rows == [], "Should return no rows when no markdown table is present."


# ---------------------------------------------------------------------------
# Tests — allowed values
# ---------------------------------------------------------------------------

def test_change_type_values_are_valid():
    rows = parse_table_rows(GOOD_OUTPUT)
    for row in rows:
        ct = row.get("Change Type", "")
        assert ct in ALLOWED_CHANGE_TYPES, (
            f"Invalid Change Type '{ct}'. Allowed: {ALLOWED_CHANGE_TYPES}"
        )


def test_practical_impact_values_are_valid():
    rows = parse_table_rows(GOOD_OUTPUT)
    for row in rows:
        pi = row.get("Practical Impact", "")
        assert pi in ALLOWED_PRACTICAL_IMPACT, (
            f"Invalid Practical Impact '{pi}'. Allowed: {ALLOWED_PRACTICAL_IMPACT}"
        )


def test_materiality_values_are_valid():
    rows = parse_table_rows(GOOD_OUTPUT)
    for row in rows:
        mat = row.get("Materiality", "")
        assert mat in ALLOWED_MATERIALITY, (
            f"Invalid Materiality '{mat}'. Allowed: {ALLOWED_MATERIALITY}"
        )


def test_modified_detected_in_bad_output():
    """Verify our parser can detect 'Modified' appearing in LLM output (so it can be rejected)."""
    rows = parse_table_rows(BAD_OUTPUT_WRONG_CHANGE_TYPE)
    has_modified = any(row.get("Change Type") == "Modified" for row in rows)
    assert has_modified, "Bad fixture should contain 'Modified' to confirm detection works."


def test_good_output_contains_no_modified():
    """Verify that well-formed output never uses 'Modified'."""
    rows = parse_table_rows(GOOD_OUTPUT)
    for row in rows:
        assert row.get("Change Type") != "Modified", (
            "'Modified' is not an allowed Change Type — must be split into Removed + Added."
        )


def test_invalid_practical_impact_detected():
    rows = parse_table_rows(BAD_OUTPUT_WRONG_IMPACT)
    invalid = [r for r in rows if r.get("Practical Impact") not in ALLOWED_PRACTICAL_IMPACT]
    assert len(invalid) > 0, "Test fixture should contain an invalid Practical Impact value."


# ---------------------------------------------------------------------------
# Tests — prompt template sanity
# ---------------------------------------------------------------------------

def test_comparison_prompt_contains_required_sections():
    from query import _COMPARISON_USER
    for section in ["TASK", "RULES", "ALLOWED VALUES", "DECISION RULES", "OUTPUT FORMAT"]:
        assert section in _COMPARISON_USER, f"Comparison prompt missing section: '{section}'"


def test_comparison_prompt_forbids_modified():
    from query import _COMPARISON_USER
    assert "Modified" in _COMPARISON_USER, (
        "Prompt must explicitly mention 'Modified' to forbid it."
    )


def test_comparison_prompt_requires_executive_summary():
    from query import _COMPARISON_USER
    assert "executive summary" in _COMPARISON_USER.lower(), (
        "Prompt must require an executive summary."
    )


def test_comparison_prompt_requires_markdown_table():
    from query import _COMPARISON_USER
    assert "markdown table" in _COMPARISON_USER.lower(), (
        "Prompt must require a markdown table output."
    )
