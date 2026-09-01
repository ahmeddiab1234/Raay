import pandas as pd

from raay.data.dialect import add_dialect_column, detect_dialect
from raay.enums.constants import Dialects


def test_egyptian_marker():
    assert detect_dialect("انا مش عارف فين ده بس عايز") == Dialects.EGYPTIAN


def test_gulf_marker():
    assert detect_dialect("وش اخبارك وليش تاخر وايد") == Dialects.GULF


def test_levantine_marker():
    assert detect_dialect("شو اخبارك بدي روح هون") == Dialects.LEVANTINE


def test_maghrebi_marker():
    assert detect_dialect("واش كاين دابا هاد البلاد") == Dialects.MAGHREBI


def test_msa_default_when_no_markers():
    assert detect_dialect("هذا منتج ممتاز وجودته عالية") == Dialects.MSA


def test_arabizi_when_latin_heavy():
    assert detect_dialect("ana mesh 3aref ya3ni eh ya a5oy") == Dialects.ARABIZI


def test_add_dialect_column():
    df = pd.DataFrame({"text": ["انا مش فاهم ايه ده", "شو هذا هيك"]})
    out = add_dialect_column(df)
    assert "dialect" in out.columns
    assert "dialect_confidence" in out.columns
    assert out["dialect"].tolist() == [Dialects.EGYPTIAN, Dialects.LEVANTINE]
