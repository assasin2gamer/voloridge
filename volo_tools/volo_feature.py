import numpy as np
import pandas as pd
import importlib
from concurrent.futures import ThreadPoolExecutor
from volo_tools import volo_feature_impl
importlib.reload(volo_feature_impl)



class VoloFeatureProcessor:
    def __init__(self, data, window=45):
        self.data = data.copy()
        self.window = window
        self.calc = volo_feature_impl.VoloFeatureImpl(window=window)

    def feature_process(self):
        df = self.data.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(["Symbol", "Date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        spy = df[df["Symbol"] == "SPY"].copy()
        spy["Log_Return"] = self.calc.calculate_log_return(spy["Adj_Close"])
        spy.dropna(subset=["Log_Return"], inplace=True)
        spy = spy[["Date", "Log_Return"]].rename(columns={"Log_Return": "SPY_Return"})
        spy.set_index("Date", inplace=True)

        symbols = df["Symbol"].unique().tolist()

        def process_symbol(symbol):
            symbol_data = df[df["Symbol"] == symbol].copy()
            symbol_data["Log_Return"] = self.calc.calculate_log_return(symbol_data["Adj_Close"])
            symbol_data.dropna(subset=["Log_Return"], inplace=True)
            symbol_data.set_index("Date", inplace=True)

            merged = symbol_data.join(spy, how="inner")

            cov = merged["Log_Return"].rolling(self.window).cov(merged["SPY_Return"])
            var_spy = merged["SPY_Return"].rolling(self.window).var()
            beta = cov / var_spy

            merged["Beta_45d"] = beta
            merged["Future_Beta_45d"] = merged["Beta_45d"].shift(-self.window)
            merged.reset_index(inplace=True)
            
            # Get variance of returns
            merged["Variance_45d"] = self.calc.calculate_variance(merged["Log_Return"])
            merged["Variance_45d_MA"] = self.calc.calculate_moving_average(merged["Variance_45d"], window=self.window)
            merged["Variance_5d_MA"] = self.calc.calculate_moving_average(merged["Variance_45d"], 5)
            
            # Get moving average of returns
            merged["MA_45d"] = self.calc.calculate_moving_average(merged["Log_Return"], window=self.window)
            merged["MA_5d"] = self.calc.calculate_moving_average(merged["Log_Return"], 5)
            merged.to_csv(f"data/{symbol}_data.csv", index=False)
            return merged

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_symbol, symbols))

        all_results = pd.concat(results, ignore_index=True)
        return all_results
