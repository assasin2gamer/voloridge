import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

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
        spy = spy.sort_index()
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

        # Forward covariance (target)
        df[f"Future_Cov_45_{stock_name}"] = (
            df["Log_Return_SPY"].shift(-future_window)
            .rolling(window=future_window)
            .cov(df[f"Log_Return_{stock_name}"].shift(-future_window))
        )

        # Fit OLS HAR-style regression on realized covariance
        y = df[f"Future_Cov_45_{stock_name}"]
        X = add_constant(df[[f"Cov_45_{stock_name}"]])
        model = OLS(y, X, missing="drop").fit()

        # Add prediction and residuals
        df[f"HAR_Pred_Cov_{stock_name}"] = model.predict(X)
        df[f"Residual_Cov_{stock_name}"] = y - df[f"HAR_Pred_Cov_{stock_name}"]

        df.dropna(inplace=True)
        return df, model