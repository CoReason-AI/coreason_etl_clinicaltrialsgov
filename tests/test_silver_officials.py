import polars as pl
import pytest

from coreason_etl_clinicaltrialsgov.transformers_polars import transform_to_silver_officials

# Mock data
STUDY_OFFICIALS = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT_OFF"},
        "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
        "contactsLocationsModule": {
            "overallOfficials": [{"name": "Dr. Smith", "role": "PI", "affiliation": "Hospital A"}]
        },
    }
}

STUDY_CONTACTS = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT_CON"},
        "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
        "contactsLocationsModule": {
            "centralContacts": [{"name": "Jane Doe", "phone": "123-456", "email": "j@d.com", "role": "CONTACT"}]
        },
    }
}

STUDY_MIXED = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT_MIX"},
        "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
        "contactsLocationsModule": {
            "overallOfficials": [{"name": "Dr. Mix", "role": "PI", "affiliation": "U"}],
            "centralContacts": [{"name": "Dr. Mix", "phone": "111", "email": "m@m.com", "role": "CONTACT"}],
        },
    }
}


@pytest.fixture
def lf_officials() -> pl.LazyFrame:
    return pl.DataFrame([STUDY_OFFICIALS]).lazy()


@pytest.fixture
def lf_contacts() -> pl.LazyFrame:
    return pl.DataFrame([STUDY_CONTACTS]).lazy()


@pytest.fixture
def lf_mixed() -> pl.LazyFrame:
    return pl.DataFrame([STUDY_MIXED]).lazy()


def test_silver_officials_overall(lf_officials: pl.LazyFrame) -> None:
    df = transform_to_silver_officials(lf_officials)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["name"] == "Dr. Smith"
    assert row["role"] == "PI"
    assert row["affiliation"] == "Hospital A"
    assert row["phone"] is None
    assert row["email"] is None


def test_silver_officials_contacts(lf_contacts: pl.LazyFrame) -> None:
    df = transform_to_silver_officials(lf_contacts)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["name"] == "Jane Doe"
    assert row["phone"] == "123-456"
    assert row["email"] == "j@d.com"
    # Role is kept as is
    assert row["role"] == "CONTACT"
    assert row["affiliation"] is None


def test_silver_officials_mixed(lf_mixed: pl.LazyFrame) -> None:
    df = transform_to_silver_officials(lf_mixed)
    # Dr. Mix appears in both lists.
    # But keys:
    # 1. Name=Dr. Mix, Role=PI, Affiliation=U
    # 2. Name=Dr. Mix, Role=CONTACT, Affiliation=None
    # These are distinct keys. So we expect 2 rows.
    assert df.height == 2

    names = df.get_column("name").to_list()
    assert names.count("Dr. Mix") == 2

    roles = df.get_column("role").to_list()
    assert "PI" in roles
    assert "CONTACT" in roles


def test_silver_officials_dedup() -> None:
    # Duplicate officials
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_DUP"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
            "contactsLocationsModule": {
                "overallOfficials": [
                    {"name": "Dr. Dup", "role": "PI", "affiliation": "A"},
                    {"name": "Dr. Dup", "role": "PI", "affiliation": "A"},
                ]
            },
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)
    assert df.height == 1
    assert df["name"][0] == "Dr. Dup"


def test_silver_officials_empty() -> None:
    data = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT_EMP"},
            "statusModule": {"studyFirstPostDateStruct": {"date": "2023-01-01"}},
        }
    }
    lf = pl.DataFrame([data]).lazy()
    df = transform_to_silver_officials(lf)
    assert df.height == 0
    # Check columns exist
    assert "name" in df.columns
    assert "affiliation" in df.columns
    assert "phone" in df.columns
