

from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from src.data.download_bhavcopy import (
    LEGACY_COLUMNS,
    download_range,
    fetch_one_day,
    parse_legacy_zip,
    parse_udiff_zip,
)


def _zip_bytes(filename: str, csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_text)
    return buf.getvalue()


def test_parse_legacy_zip_roundtrip():
    csv_text = (
        "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
        "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP\n"
        "OPTIDX,NIFTY,26-SEP-2024,24000,CE,100,110,90,95,95,500,1000.5,20000,300,02-SEP-2024\n"
    )
    content = _zip_bytes("fo02SEP2024bhav.csv", csv_text)
    df = parse_legacy_zip(content)
    assert list(df.columns) == LEGACY_COLUMNS
    assert df.iloc[0]["SYMBOL"] == "NIFTY"
    assert df.iloc[0]["STRIKE_PR"] == 24000


def test_parse_udiff_zip_maps_to_legacy_schema():
    csv_text = (
        "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
        "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,"
        "PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,"
        "TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
        "2024-08-14,2024-08-14,FO,NSE,IDO,36742,,NIFTY,,2024-08-22,2024-08-22,24350.00,CE,"
        "NIFTY2482224350CE,129.40,133.25,85.80,93.85,94.00,125.45,24143.75,93.85,622200,208700,"
        "113799,69558588098.75,32711,F1,25,,,,,\n"
    )
    content = _zip_bytes("BhavCopy_NSE_FO_0_0_0_20240814_F_0000.csv", csv_text)
    df = parse_udiff_zip(content)

    assert list(df.columns) == LEGACY_COLUMNS
    row = df.iloc[0]
    assert row["INSTRUMENT"] == "OPTIDX"  # IDO -> OPTIDX
    assert row["SYMBOL"] == "NIFTY"
    assert row["EXPIRY_DT"] == "22-AUG-2024"
    assert row["STRIKE_PR"] == 24350.00
    assert row["OPTION_TYP"] == "CE"
    assert row["SETTLE_PR"] == pytest.approx(93.85)
    assert row["VAL_INLAKH"] == pytest.approx(69558588098.75 / 100_000.0)
    assert row["TIMESTAMP"] == "14-AUG-2024"


def test_fetch_one_day_prefers_legacy_over_udiff(monkeypatch):
    calls = []

    def fake_get(url, max_retries=3, backoff_seconds=3.0):
        calls.append(url)
        if "historical" in url:
            return type("R", (), {"content": _zip_bytes("x.csv", "INSTRUMENT\nOPTIDX\n")})()
        raise AssertionError("should not fall back to UDiFF when legacy succeeds")

    monkeypatch.setattr("src.data.download_bhavcopy._get_with_retry", fake_get)
    monkeypatch.setattr("src.data.download_bhavcopy.parse_legacy_zip", lambda c: pd.DataFrame({"INSTRUMENT": ["OPTIDX"]}))

    result = fetch_one_day(date(2020, 9, 1))
    assert result is not None
    assert len(calls) == 1
    assert "historical" in calls[0]


def test_fetch_one_day_falls_back_to_udiff(monkeypatch):
    def fake_get(url, max_retries=3, backoff_seconds=3.0):
        if "historical" in url:
            return None  # legacy 404s
        return type("R", (), {"content": b"udiff-bytes"})()

    monkeypatch.setattr("src.data.download_bhavcopy._get_with_retry", fake_get)
    monkeypatch.setattr("src.data.download_bhavcopy.parse_udiff_zip", lambda c: pd.DataFrame({"INSTRUMENT": ["FUTIDX"]}))

    result = fetch_one_day(date(2024, 8, 14))
    assert result is not None
    assert result.iloc[0]["INSTRUMENT"] == "FUTIDX"


def test_fetch_one_day_returns_none_when_both_formats_404(monkeypatch):
    monkeypatch.setattr("src.data.download_bhavcopy._get_with_retry", lambda url, max_retries=3, backoff_seconds=3.0: None)
    assert fetch_one_day(date(2024, 8, 15)) is None  # e.g. Independence Day holiday


def test_download_range_is_idempotent(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def fake_fetch(d):
        call_count["n"] += 1
        return pd.DataFrame({"INSTRUMENT": ["OPTIDX"], "SYMBOL": ["NIFTY"]})

    monkeypatch.setattr("src.data.download_bhavcopy.fetch_one_day", fake_fetch)

    start, end = date(2024, 9, 2), date(2024, 9, 4)  # Mon, Tue, Wed -> 3 weekdays
    result1 = download_range(start, end, tmp_path, sleep_seconds=0)
    assert result1["downloaded"] == 3
    assert call_count["n"] == 3

    result2 = download_range(start, end, tmp_path, sleep_seconds=0)
    assert result2["already_present"] == 3
    assert result2["downloaded"] == 0
    assert call_count["n"] == 3  # no new network calls on the second pass


def test_download_range_skips_weekends(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.data.download_bhavcopy.fetch_one_day", lambda d: pd.DataFrame({"INSTRUMENT": ["OPTIDX"]})
    )
    # 2024-09-07 (Sat) to 2024-09-08 (Sun): pure weekend, zero weekday iterations
    result = download_range(date(2024, 9, 7), date(2024, 9, 8), tmp_path, sleep_seconds=0)
    assert result == {"downloaded": 0, "already_present": 0, "holidays": 0}


def test_download_range_counts_holidays(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.download_bhavcopy.fetch_one_day", lambda d: None)
    result = download_range(date(2024, 8, 15), date(2024, 8, 15), tmp_path, sleep_seconds=0)  # Independence Day
    assert result["holidays"] == 1
    assert not (tmp_path / "2024-08-15.parquet").exists()
