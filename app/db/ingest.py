"""Pure, unit-tested parsing functions for the CA purchase order CSV.

Field quirks (verified against the full 346,018-row dataset, not assumed):
- Supplier Qualifications is SPACE-separated, not comma-separated.
- Location has three shapes: "<zip>\\n(<lat>, <lon>)", "<zip+4>\\n(<lat>, <lon>)",
  or "<zip>\\n" with no coordinates (5.2% of non-empty values — geocoding
  failed upstream). lat/lon must be treated as independently optional.
- Dates are 100% uniform MM/DD/YYYY strings — no fallback parsing needed.
"""
from __future__ import annotations

import re
from datetime import datetime

_LOCATION_RE = re.compile(r"^(?P<zip>[\d-]+)\n(?:\((?P<lat>-?\d+\.\d+), (?P<lon>-?\d+\.\d+)\))?$")


def parse_price(raw: str) -> float | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    cleaned = raw.replace("$", "").replace(",", "")
    return float(cleaned)


def parse_qualifications(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return raw.split()


def parse_location(raw: str) -> dict:
    raw = (raw or "").strip("\n")
    if not raw:
        return {"zip": None, "lat": None, "lon": None}
    match = _LOCATION_RE.match(raw + "\n" if not raw.endswith("\n") else raw)
    if not match:
        # Fallback: treat the whole thing as a zip if it doesn't match the
        # expected shape rather than raising — malformed rows should not
        # crash ingestion of the other 345,999 rows.
        return {"zip": raw.split("\n")[0] or None, "lat": None, "lon": None}
    lat = match.group("lat")
    lon = match.group("lon")
    return {
        "zip": match.group("zip") or None,
        "lat": float(lat) if lat else None,
        "lon": float(lon) if lon else None,
    }


def parse_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    return datetime.strptime(raw, "%m/%d/%Y")


def derive_quarter(creation_date: datetime) -> int:
    """CA state fiscal year: Q1=Jul-Sep, Q2=Oct-Dec, Q3=Jan-Mar, Q4=Apr-Jun."""
    month = creation_date.month
    if month in (7, 8, 9):
        return 1
    if month in (10, 11, 12):
        return 2
    if month in (1, 2, 3):
        return 3
    return 4


def clean_document(raw_row: dict) -> dict:
    creation_date = parse_date(raw_row["Creation Date"])
    return {
        "creation_date": creation_date,
        "purchase_date": parse_date(raw_row["Purchase Date"]),
        "fiscal_year": raw_row["Fiscal Year"] or None,
        "quarter": derive_quarter(creation_date) if creation_date else None,
        "lpa_number": raw_row["LPA Number"] or None,
        "purchase_order_number": raw_row["Purchase Order Number"] or None,
        "requisition_number": raw_row["Requisition Number"] or None,
        "acquisition_type": raw_row["Acquisition Type"] or None,
        "sub_acquisition_type": raw_row["Sub-Acquisition Type"] or None,
        "acquisition_method": raw_row["Acquisition Method"] or None,
        "sub_acquisition_method": raw_row["Sub-Acquisition Method"] or None,
        "department_name": raw_row["Department Name"] or None,
        "supplier_code": raw_row["Supplier Code"] or None,
        "supplier_name": raw_row["Supplier Name"] or None,
        "supplier_qualifications": parse_qualifications(raw_row["Supplier Qualifications"]),
        "supplier_zip_code": raw_row["Supplier Zip Code"] or None,
        "calcard": raw_row["CalCard"] == "YES",
        "item_name": raw_row["Item Name"] or None,
        "item_description": raw_row["Item Description"] or None,
        "quantity": float(raw_row["Quantity"]) if raw_row["Quantity"] else None,
        "unit_price": parse_price(raw_row["Unit Price"]),
        "total_price": parse_price(raw_row["Total Price"]),
        "classification_codes": raw_row["Classification Codes"] or None,
        "normalized_unspsc": raw_row["Normalized UNSPSC"] or None,
        "commodity_title": raw_row["Commodity Title"] or None,
        "class_code": raw_row["Class"] or None,
        "class_title": raw_row["Class Title"] or None,
        "family": raw_row["Family"] or None,
        "family_title": raw_row["Family Title"] or None,
        "segment": raw_row["Segment"] or None,
        "segment_title": raw_row["Segment Title"] or None,
        "location": parse_location(raw_row["Location"]),
    }
