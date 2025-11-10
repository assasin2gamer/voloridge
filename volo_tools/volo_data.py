import importlib
import numpy as np
import pandas as pd
from volo_tools import volo_feature
importlib.reload(volo_feature)

class VoloDataProcessor:
    def __init__(self):
        self.data = None

    def process(self, data_path):
        """
        CSV has: Code,Symbol,Date,Close,Volume,Adjustment Factor
        Separate by Symbol and handle VXX splits
        """
        # Load and ensure all dates are parsed as datetime
        self.data = pd.read_csv(data_path)
        self.data["Date"] = pd.to_datetime(self.data["Date"], format="%m/%d/%Y", errors="coerce")
        self.data.sort_values(by=["Symbol", "Date"], inplace=True)

        # Compute initial adjusted close
        self.data["Adj_Close"] = self.data["Close"] * self.data["Adjustment Factor"]

        # Print which symbols have adjustment changes
        adjustment_issues = self.data[self.data["Adjustment Factor"] != 1]
        if not adjustment_issues.empty:
            adjustment_counts = adjustment_issues.groupby("Symbol").size()
            print("Stocks with adjustment factor changes:")
            print(adjustment_counts)

        # List of all symbols
        symbols = self.data["Symbol"].unique()

        for symbol in symbols:
            symbol_data = self.data[self.data["Symbol"] == symbol].copy()
            symbol_data.sort_values(by="Date", inplace=True)

            if symbol == "VXX":
                # Define all reverse split events
                vxx_splits = pd.DataFrame({
                    "Date": pd.to_datetime([
                        "2010-11-09", "2012-10-05", "2013-11-08",
                        "2016-08-09", "2017-08-23", "2021-04-23",
                        "2023-03-07", "2024-07-24"
                    ]),
                    "Split Ratio": [0.25] * 8
                }).sort_values("Date")

                # Apply each split only once forward (non-cumulative)
                for _, row in vxx_splits.iterrows():
                    split_date = row["Date"]
                    split_ratio = row["Split Ratio"]
                    # Adjust only VXX data after that split date
                    self.data.loc[
                        (self.data["Symbol"] == "VXX") & (self.data["Date"] >= split_date),
                        "Adjustment Factor"
                    ] *= split_ratio

            # Write per-symbol CSV
            symbol_data.to_csv(f"data/{symbol}_data.csv", index=False)

        # Recalculate Adj_Close after all splits
        self.data["Adj_Close"] = self.data["Close"] * self.data["Adjustment Factor"]

        # Save date ranges for sanity check
        date_ranges = (
            self.data.groupby("Symbol")["Date"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={"min": "Start Date", "max": "End Date"})
        )
        date_ranges.to_csv("data/date_ranges.csv", index=False)

        # Feed adjusted data into feature processor
        result = volo_feature.VoloFeatureProcessor(self.data).feature_process()
        return result
