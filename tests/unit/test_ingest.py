from datetime import datetime

from app.db.ingest import (
    clean_document,
    derive_quarter,
    parse_date,
    parse_location,
    parse_price,
    parse_qualifications,
)


def test_parse_price_strips_dollar_and_commas():
    assert parse_price("$1,234.56") == 1234.56


def test_parse_price_handles_negative_credit():
    assert parse_price("-$30,861,228.00") == -30861228.00


def test_parse_price_empty_string_is_none():
    assert parse_price("") is None


def test_parse_qualifications_splits_on_whitespace_not_comma():
    # Real value from the dataset — this field is space-separated, NOT
    # comma-separated. A prior version of this parser got this wrong.
    assert parse_qualifications("CA-MB CA-SB CA-SBE") == ["CA-MB", "CA-SB", "CA-SBE"]


def test_parse_qualifications_empty_is_empty_list():
    assert parse_qualifications("") == []


def test_parse_location_full_with_coordinates():
    raw = "95841\n(38.662263, -121.346136)"
    assert parse_location(raw) == {"zip": "95841", "lat": 38.662263, "lon": -121.346136}


def test_parse_location_zip_plus_four_with_coordinates():
    raw = "95812-1433\n(38.582087, -121.50012)"
    assert parse_location(raw) == {"zip": "95812-1433", "lat": 38.582087, "lon": -121.50012}


def test_parse_location_zip_only_no_coordinates():
    # Real value — geocoding failed upstream for this row; only the zip
    # and a trailing newline remain. lat/lon must be None, not an error.
    raw = "98733\n"
    assert parse_location(raw) == {"zip": "98733", "lat": None, "lon": None}


def test_parse_location_empty_string():
    assert parse_location("") == {"zip": None, "lat": None, "lon": None}


def test_parse_date_standard_format():
    assert parse_date("08/27/2013") == datetime(2013, 8, 27)


def test_derive_quarter_ca_fiscal_year_q1_is_july_through_september():
    assert derive_quarter(datetime(2013, 8, 27)) == 1


def test_derive_quarter_ca_fiscal_year_q4_is_april_through_june():
    assert derive_quarter(datetime(2014, 5, 1)) == 4


def test_derive_quarter_ca_fiscal_year_q3_is_january_through_march():
    assert derive_quarter(datetime(2014, 1, 29)) == 3


def test_clean_document_produces_expected_shape():
    raw_row = {
        "Creation Date": "08/27/2013",
        "Purchase Date": "",
        "Fiscal Year": "2013-2014",
        "LPA Number": "7-12-70-26",
        "Purchase Order Number": "REQ0011118",
        "Requisition Number": "REQ0011118",
        "Acquisition Type": "IT Goods",
        "Sub-Acquisition Type": "",
        "Acquisition Method": "WSCA/Coop",
        "Sub-Acquisition Method": "",
        "Department Name": "Consumer Affairs, Department of",
        "Supplier Code": "1740272",
        "Supplier Name": "Pitney Bowes",
        "Supplier Qualifications": "",
        "Supplier Zip Code": "",
        "CalCard": "NO",
        "Item Name": "USB",
        "Item Description": "USB",
        "Quantity": "1",
        "Unit Price": "$1.00",
        "Total Price": "$1.00",
        "Classification Codes": "",
        "Normalized UNSPSC": "",
        "Commodity Title": "",
        "Class": "",
        "Class Title": "",
        "Family": "",
        "Family Title": "",
        "Segment": "",
        "Segment Title": "",
        "Location": "",
    }
    doc = clean_document(raw_row)
    assert doc["creation_date"] == datetime(2013, 8, 27)
    assert doc["fiscal_year"] == "2013-2014"
    assert doc["quarter"] == 1
    assert doc["total_price"] == 1.00
    assert doc["quantity"] == 1.0
    assert doc["supplier_qualifications"] == []
    assert doc["location"] == {"zip": None, "lat": None, "lon": None}
