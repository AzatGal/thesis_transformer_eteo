
_base_ = [
    f"../../_base_/data_preprocessor/yahoofinance/dj30.py",
]

data = dict(type="YfinancePreprocessor",
            data_path="workdir/sp500",
            train_path="workdir/sp500/train.csv",
            valid_path="workdir/sp500/valid.csv",
            test_path="workdir/sp500/test.csv",
            start_date = "2000-01-01",
            end_date = "2019-01-01",
            train_valid_test_portion = [0.8,0.1,0.1],
            tickers=[
                "AAPL",
                "MSFT",
                "JPM",
                "V",
                "RTX",
                "PG",
                "GS",
                "NKE",
                "DIS",
                "AXP",
                "HD",
                "INTC",
                "WMT",
                "IBM",
                "MRK",
                "UNH",
                "KO",
                "CAT",
                "TRV",
                "JNJ",
                "CVX",
                "MCD",
                "VZ",
                "CSCO",
                "XOM",
                "BA",
                "MMM",
                "PFE",
                "WBA",
                "DD",
            ],
)