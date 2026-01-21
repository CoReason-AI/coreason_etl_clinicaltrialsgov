import polars as pl

from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_officials

# --- Complex Scenarios ---


def test_silver_officials_unicode() -> None:
    # Emoji and RTL
    name = "Dr. 😊"
    role = "עוזר מחקר"  # Hebrew for Research Assistant
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_UNI"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {"overallOfficials": [{"name": name, "role": role, "affiliation": "A"}]},
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)
    assert df.height == 1
    assert df["name"][0] == name
    assert df["role"][0] == role


def test_silver_officials_whitespace_dedup() -> None:
    # " Name " vs "Name" should dedup if stripped correctly
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_WS"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {
                "overallOfficials": [
                    {"name": " Dr. Space ", "role": "PI", "affiliation": "A"},
                    {"name": "Dr. Space", "role": "PI", "affiliation": "A"},
                ]
            },
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)
    # Should be 1 record
    assert df.height == 1
    assert df["name"][0] == "Dr. Space"


def test_silver_officials_mixed_types() -> None:
    # Integer as name
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_TYPE"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {"overallOfficials": [{"name": 123, "role": 999, "affiliation": "A"}]},
        }
    }
    # Create with schema relaxation if needed, but Polars.DataFrame usually handles mixed types well in lists
    lf = pl.DataFrame([data], strict=False).lazy()
    df = transform_to_silver_officials(lf)
    assert df.height == 1
    assert df["name"][0] == "123"
    assert df["role"][0] == "999"


def test_silver_officials_delimiter_collision() -> None:
    # Name with pipe
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_PIPE"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {
                "overallOfficials": [
                    {"name": "A|B", "role": "R", "affiliation": "A"},
                    {"name": "A_B", "role": "R", "affiliation": "A"},
                ]
            },
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)

    # ID generation replaces | with _
    # So "A|B" -> "A_B".
    # "A_B" -> "A_B".
    # This results in collision.
    # We expect 1 record due to collision resolution (first wins).
    assert df.height == 1
    # First one was "A|B"
    assert df["name"][0] == "A|B"


def test_silver_officials_cross_source_dedup() -> None:
    # Same person in Officials and Contacts
    # Key: Name, Role, Affiliation.
    # Contact: Name, Role, Affiliation=None.
    # If official has Affiliation=None (missing), do they merge?
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_CROSS"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {
                "overallOfficials": [{"name": "Dr. Same", "role": "PI", "affiliation": None}],
                "centralContacts": [{"name": "Dr. Same", "role": "PI", "phone": "123"}],
            },
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)

    # Official: Name="Dr. Same", Role="PI", Affil=None.
    # Contact: Name="Dr. Same", Role="PI", Affil=None (literal).
    # Keys should be identical.
    # Expect 1 record.
    assert df.height == 1
    # Code processes Officials THEN Contacts.
    # Concat Order: [Officials, Contacts].
    # Keep First -> Keep Official.
    # So Phone is None.
    assert df["name"][0] == "Dr. Same"
    assert df["phone"][0] is None
