

from datetime import date

import pandas as pd

from src.data.clean_nifty import extract_raw, extract_raw_daily_dir, merge_spot_and_rate


def _write_minimal_bhavcopy_csv(path):
    path.write_text(
        "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
        "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP\n"
        "OPTIDX,NIFTY,30-JAN-2020,12000,CE,100,110,90,105,105,500,1000.5,20000,300,02-JAN-2020\n"
        "FUTIDX,NIFTY,30-JAN-2020,0,XX,12050,12060,12040,12055,12055,1000,2000.5,50000,300,02-JAN-2020\n"
    )


def test_extract_raw_returns_plain_date_not_timestamp(tmp_path):
    csv_path = tmp_path / "fobhav.csv"
    _write_minimal_bhavcopy_csv(csv_path)

    df = extract_raw(csv_path, "NIFTY", 2000, 2025)
    assert len(df) == 2
    assert isinstance(df["trade_date"].iloc[0], date)
    assert not isinstance(df["trade_date"].iloc[0], pd.Timestamp)
    assert isinstance(df["expiry_date"].iloc[0], date)
    assert not isinstance(df["expiry_date"].iloc[0], pd.Timestamp)


def test_extract_raw_and_extract_raw_daily_dir_dtypes_match(tmp_path):
    csv_path = tmp_path / "fobhav.csv"
    _write_minimal_bhavcopy_csv(csv_path)
    kaggle = extract_raw(csv_path, "NIFTY", 2000, 2025)

    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(
        {
            "INSTRUMENT": ["OPTIDX"], "SYMBOL": ["NIFTY"], "EXPIRY_DT": ["27-FEB-2020"], "STRIKE_PR": [12100.0],
            "OPTION_TYP": ["CE"], "OPEN": [90.0], "HIGH": [95.0], "LOW": [85.0], "CLOSE": [92.0], "SETTLE_PR": [92.0],
            "CONTRACTS": [400], "VAL_INLAKH": [50.0], "OPEN_INT": [10000], "CHG_IN_OI": [100], "TIMESTAMP": ["03-FEB-2020"],
        }
    ).to_parquet(daily_dir / "2020-02-03.parquet", index=False)
    daily = extract_raw_daily_dir(daily_dir, "NIFTY")

    assert type(kaggle["trade_date"].iloc[0]) is type(daily["trade_date"].iloc[0])
    assert type(kaggle["expiry_date"].iloc[0]) is type(daily["expiry_date"].iloc[0])

    # concatenating must not silently coerce into an unmatchable mixed-type column
    combined = pd.concat([kaggle, daily], ignore_index=True)
    assert all(isinstance(d, date) and not isinstance(d, pd.Timestamp) for d in combined["trade_date"])


def test_merge_spot_and_rate_matches_kaggle_sourced_dates(tmp_path):
    csv_path = tmp_path / "fobhav.csv"
    _write_minimal_bhavcopy_csv(csv_path)
    raw = extract_raw(csv_path, "NIFTY", 2000, 2025)
    options = raw[raw["INSTRUMENT"] == "OPTIDX"].copy()
    options["forward"] = 12050.0  # stand in for the futures-merged forward

    spot = pd.Series({date(2020, 1, 2): 12040.0}, name="spot")
    merged = merge_spot_and_rate(options, spot)

    assert len(merged) == 1  # would be 0 if trade_date types didn't match the spot index
    assert merged["spot"].iloc[0] == 12040.0
