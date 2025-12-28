# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_clinicaltrialsgov

from typing import Any

from coreason_etl_clinicaltrialsgov.transformers import transform_study


def test_silver_reference_strict_exclusion_of_see_also_links() -> None:
    """
    Strictly verify that 'seeAlsoLinks' are NOT transformed into SilverReferences.
    Only 'references' should be included.
    """
    raw_study: dict[str, Any] = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_STRICT_TEST"},
            "referencesModule": {
                "references": [
                    {"pmid": "111", "citation": "Valid Ref 1", "type": "BACKGROUND"},
                ],
                "seeAlsoLinks": [
                    {"label": "Link 1", "url": "http://example.com"},
                    {"label": "Link 2", "url": "http://example.org"},
                ],
            },
        }
    }

    result = transform_study(raw_study)
    refs = result.get("silver_references", [])

    # 1. Verify we have exactly 1 reference (from 'references')
    assert len(refs) == 1

    # 2. Verify the content of that reference
    ref = refs[0]
    assert ref["pmid"] == "111"
    assert ref["citation"] == "Valid Ref 1"

    # 3. Explicitly verify no seeAlsoLinks data leaked in
    # This is slightly redundant with len=1, but ensures we didn't somehow map seeAlsoLinks INSTEAD of references
    # or mix them.
    for r in refs:
        # Check against seeAlsoLink values
        assert r.get("label") != "Link 1"
        assert r.get("url") != "http://example.com"
