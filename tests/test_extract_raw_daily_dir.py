
import pandas as pd

from src.data.clean_nifty import extract_raw_daily_dir


def test_extract_raw_daily_dir_parses_and_filters(tmp_path):
    df = pd.DataFrame(
        {
            "INSTRUMENT": ["OPTIDX", "OPTIDX", "FUTIDX", "OPTIDX"],
            "SYMBOL": ["NIFTY", "NIFTY", "NIFTY", "BANKNIFTY"],
            "EXPIRY_DT": ["26-SEP-2024", "26-SEP-2024", "26-SEP-2024", "26-SEP-2024"],
            "STRIKE_PR": [24000.0, 24000.0, 0.0, 50000.0],
            "OPTION_TYP": ["CE", "PE", "XX", "CE"],
            "OPEN": [100.0, 90.0, 24100.0, 200.0],
            "HIGH": [110.0, 95.0, 24200.0, 210.0],
            "LOW": [95.0, 85.0, 24000.0, 190.0],
            "CLOSE": [105.0, 92.0, 24150.0, 205.0],
            "SETTLE_PR": [105.0, 92.0, 24150.0, 205.0],
            "CONTRACTS": [500, 300, 1000, 50],
            "VAL_INLAKH": [100.0, 50.0, 200.0, 10.0],
            "OPEN_INT": [20000, 15000, 50000, 5000],
            "CHG_IN_OI": [100, 50, 200, 10],
            "TIMESTAMP": ["02-SEP-2024", "02-SEP-2024", "02-SEP-2024", "02-SEP-2024"],
        }
    )
    df.to_parquet(tmp_path / "2024-09-02.parquet", index=False)

    result = extract_raw_daily_dir(tmp_path, "NIFTY")
    assert len(result) == 3  # BANKNIFTY row filtered out
    assert set(result["INSTRUMENT"]) == {"OPTIDX", "FUTIDX"}
    assert result.iloc[0]["trade_date"].isoformat() == "2024-09-02"
    assert result.iloc[0]["expiry_date"].isoformat() == "2024-09-26"
    assert result.iloc[0]["strike"] == 24000.0


def test_extract_raw_daily_dir_empty_returns_empty_frame(tmp_path):
    result = extract_raw_daily_dir(tmp_path, "NIFTY")
    assert result.empty
    assert "trade_date" in result.columns


def test_extract_raw_daily_dir_missing_dir_returns_empty_frame(tmp_path):
    result = extract_raw_daily_dir(tmp_path / "does_not_exist", "NIFTY")
    assert result.empty
