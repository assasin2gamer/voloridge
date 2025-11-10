import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
import os
import glob
class VoloFeatureImpl:
    def __init__(self, window=45):
        self.window = window

    def calculate_volatility(self, returns):
        return returns.rolling(self.window).std()

    def calculate_log_return(self, prices):
        return np.log(prices / prices.shift(1))
            
    def calculate_beta_coefficient(self, symbol_returns, spy_returns):
        covariance = symbol_returns.rolling(self.window).cov(spy_returns)
        spy_variance = spy_returns.rolling(self.window).var()
        return covariance / spy_variance
    
    def calculate_hurst_exponent(self, time_series):
        lags = range(2, 20)
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def calculate_variance(self, returns):
        return returns.rolling(window=self.window).var()

    def calculate_moving_average(self, returns, window):
        return returns.rolling(window=window).mean()
    
    def _prepare_spy_baseline(self, spy: pd.DataFrame, future_window: int = 45) -> pd.DataFrame:
        # Sort by date
        # Check spy index
        spy = spy.sort_index()
        print(spy.head())
        spy["RV_future"] = (
            spy["Log_Return"]
            .shift(-future_window)                # look forward
            .rolling(window=future_window)        # take variance over the *future* window
            .var()
        )        
        y = spy["RV_future"]
        spy['RQ_daily'] = spy['Log_Return']**4
        spy['RV_x_RQ'] = spy['Variance_45d'] * np.sqrt(spy['RQ_daily'])
        X = add_constant(spy[["Variance_45d", "Variance_45d_MA", "Variance_5d_MA"]])
        har_model = OLS(y, X, missing="drop").fit()
        spy["HAR_Pred"] = har_model.predict(X)
        spy["Residual"] = spy["RV_future"] - spy["HAR_Pred"]
        return spy, har_model


    def _prepare_spy_baseline_covar(
        self, spy: pd.DataFrame, stock: pd.DataFrame, stock_name: str, future_window: int = 45
    ) -> pd.DataFrame:
        """
        Prepares 45-day forward realized covariance between SPY and another stock.
        Produces a suffix-specific dataframe with both Cov_45_<stock> and Future_Cov_45_<stock>.
        """

        spy = spy.sort_index()
        stock = stock.sort_index()

        df = pd.DataFrame(index=spy.index)
        df["Log_Return_SPY"] = spy["Log_Return"]
        df[f"Log_Return_{stock_name}"] = stock["Log_Return"]

        # Rolling covariance
        df[f"Cov_45_{stock_name}"] = (
            df["Log_Return_SPY"].rolling(window=future_window).cov(df[f"Log_Return_{stock_name}"])
        )
        
        # Rolling variance of stock
        df[f"Var_45_{stock_name}"] = (
            df[f"Log_Return_{stock_name}"].rolling(window=future_window).var()
        )
        
        # Rolling correlation
        df[f"Corr_45_{stock_name}"] = (
            df["Log_Return_SPY"].rolling(window=future_window).corr(df[f"Log_Return_{stock_name}"])
        )

        # Forward covariance (target)
        df[f"Future_Cov_45_{stock_name}"] = (
            df["Log_Return_SPY"].shift(-future_window)
            .rolling(window=future_window)
            .cov(df[f"Log_Return_{stock_name}"].shift(-future_window))
        )
        
        # Foward correlation (target)
        df[f"Future_Corr_45_{stock_name}"] = (
            df["Log_Return_SPY"].shift(-future_window)
            .rolling(window=future_window)
            .corr(df[f"Log_Return_{stock_name}"].shift(-future_window))
        )

        # Fit OLS HAR-style regression on future correlation
        y = df[f"Future_Corr_45_{stock_name}"]
        X = add_constant(df[[f"Corr_45_{stock_name}"]])
        model = OLS(y, X, missing="drop").fit()
        # Add prediction and residuals
        df[f"HAR_Pred_Corr_{stock_name}"] = model.predict(X)
        df["target_correlation"] = df[f"Future_Corr_45_{stock_name}"]/df[f"HAR_Pred_Corr_{stock_name}"]

        # Fit OLS HAR-style regression on stock variance
        y_var = stock["Variance_45d"].shift(-future_window).rolling(window=future_window).var()
        X_var = add_constant(stock[["Variance_45d", "Variance_45d_MA", "Variance_5d_MA"]])
        model_var = OLS(y_var, X_var, missing="drop").fit()
        df["target_variance"] = df[f"Var_45_{stock_name}"].shift(-45).rolling(window=45).var()/df[f"HAR_Pred_Var_{stock_name}"]

        # Add prediction and residuals
        df[f"HAR_Pred_Var_{stock_name}"] = model_var.predict(X_var)
        print(df.head())
        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)
        
        return df, model
    
    def covar_features(self, stock_name):
        # Load base datasets
        spy_data = pd.read_csv("data/SPY_data.csv", parse_dates=["Date"])
        spy_data.set_index("Date", inplace=True)

        stock_data = pd.read_csv(f"data/{stock_name}_data.csv", parse_dates=["Date"])
        stock_data.set_index("Date", inplace=True)

        # Separate covariances
        spy_stock_df, model_stock = self._prepare_spy_baseline_covar(
            spy_data, stock_data, stock_name, future_window=45
        )
        csv_files = glob.glob(os.path.join("data/", "*.csv"))

        valid_symbols = [os.path.basename(f).replace("_data.csv", "") for f in csv_files if pd.read_csv(f).shape[0] > 1300]
        features = self.feature_collector(valid_symbols, spy_stock_df)
        # Merge features back
        spy_stock_df = spy_stock_df.merge(
            features,
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", f"_feat")
        )
        

        # Target variable = Future Correlation with stock
        spy_stock_df["target_correlation"] = spy_stock_df[f"Future_Corr_45_{stock_name}"]/spy_stock_df[f"HAR_Pred_Corr_{stock_name}"]
        # Target variable = Future Variance of stock
        spy_stock_df["target_variance"] = spy_stock_df[f"Var_45_{stock_name}"].shift(-45).rolling(window=45).var()/spy_stock_df[f"HAR_Pred_Var_{stock_name}"]
        spy_stock_df.dropna(inplace=True)
        return spy_stock_df, model_stock
    
    def feature_collector(self, all_symbols, spy):
        data = spy.copy()

        for symbol in all_symbols:
            file_path = f"data/{symbol}_data.csv"
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping.")
                continue

            symbol_data = pd.read_csv(file_path)
            symbol_data = symbol_data.drop(columns=["Future_Beta_45d"], errors="ignore")
            # Set date
            # merge on index
            data = pd.merge(
                data,
                symbol_data,
                left_index=True,
                right_index=True,
                how="outer",
                suffixes=("", f"_{symbol}")
            )

        #remove Future_Beta_45d
        data = data.loc[:, ~data.columns.str.contains('Future_Beta_45d')]
        data = data.loc[:, ~data.columns.str.contains('Code')]
        data = data.loc[:, ~data.columns.str.contains('RV_future')]
        data = data.loc[:, ~data.columns.str.contains('Symbol')]
        data = data.loc[:, ~data.columns.str.contains('Residual')]
        data.dropna(inplace=True)
        # set index to Date
        print("Final feature set preview:")
        data.set_index("Date", inplace=True)
        print(data.head())
        return data
