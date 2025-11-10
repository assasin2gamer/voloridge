import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
from arch import arch_model # Added for GARCH modeling
import warnings

# Suppress warnings from the GARCH model fitting process
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MultiAssetVolatilityForecaster:
    """
    A class to forecast volatility using and comparing three different methods:
    1. A hybrid HAR-XGBoost model.
    2. A standard GARCH(1,1) model.
    3. A naive Random Walk (Persistence) model.
    """
    def __init__(self, spy_path: str = "data/SPY_data.csv", data_dir: str = "data"):
        self.spy_path = spy_path
        self.data_dir = data_dir
        if not os.path.exists(self.spy_path):
            raise FileNotFoundError(f"SPY data not found at {self.spy_path}. Please check the path.")

    # ------------------------------------------------------------
    # Step 1: Data Calculation Functions
    # ------------------------------------------------------------
    def _calculate_daily_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Scale returns by 100 for better GARCH model stability
        df['log_return'] = np.log(df['Adj_Close'] / df['Adj_Close'].shift(1)) * 100
        df['RV_daily'] = df['log_return']**2
        df['RQ_daily'] = df['log_return']**4
        rolling_rv_mean = df['RV_daily'].rolling(window=22).mean()
        df['Jumps'] = np.maximum(df['RV_daily'] - rolling_rv_mean.shift(1), 0)
        rolling_std = df['log_return'].rolling(window=22).std()
        df['Jumps'] = df['Jumps'].where(df['log_return'].abs() > 2.5 * rolling_std.shift(1), 0)
        return df.dropna(subset=['log_return', 'RV_daily'])

    # ------------------------------------------------------------
    # Step 2: Prepare SPY Baseline (HAR Model)
    # ------------------------------------------------------------
    def _prepare_spy_baseline(self, spy_df: pd.DataFrame, future_window: int = 45) -> pd.DataFrame:
        spy = self._calculate_daily_measures(spy_df)
        spy["RV_weekly"] = spy["RV_daily"].rolling(5).mean()
        spy["RV_monthly"] = spy["RV_daily"].rolling(22).mean()
        spy["RV_future"] = spy["RV_daily"].rolling(window=future_window).mean().shift(-future_window)
        spy.dropna(subset=["RV_daily", "RV_weekly", "RV_monthly", "RV_future"], inplace=True)

        if spy.empty:
            raise ValueError("The SPY DataFrame is empty after calculations.")
        
        X = add_constant(spy[["RV_daily", "RV_weekly", "RV_monthly"]])
        y = spy["RV_future"]
        har_model = OLS(y, X).fit()
        spy["HAR_Pred"] = har_model.predict(X)
        return spy

    # ------------------------------------------------------------
    # Step 3: Build Features from Other Assets
    # ------------------------------------------------------------
    def _build_asset_features(self, asset_dfs_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        all_features = []
        for ticker, df in asset_dfs_map.items():
            # remove ['Future_Beta_45d'] if exists
            if 'Future_Beta_45d' in df.columns:
                df = df.drop(columns=['Future_Beta_45d'])
            df_with_measures = self._calculate_daily_measures(df)
            df_with_measures[f"{ticker}_mom20"] = df_with_measures["Adj_Close"].pct_change(20)
            df_with_measures[f"{ticker}_mom60"] = df_with_measures["Adj_Close"].pct_change(60)
            df_with_measures[f"{ticker}_vol_of_vol"] = df_with_measures["RV_daily"].rolling(22).std()
            all_features.append(df_with_measures[["Date", f"{ticker}_mom20", f"{ticker}_mom60", f"{ticker}_vol_of_vol"]])
        
        if not all_features: raise ValueError("No valid asset data to build features from.")
        merged_df = all_features[0]
        for df in all_features[1:]:
            merged_df = pd.merge(merged_df, df, on="Date", how="inner")
        return merged_df.dropna()

    # ------------------------------------------------------------
    # Step 4: Forecasting Models
    # ------------------------------------------------------------
    def _rolling_forecast_xgb(self, df: pd.DataFrame, feature_cols: List[str], train_window: int = 500, step: int = 30) -> pd.DataFrame:
        df["target_multiplier"] = df["RV_future"] / df["HAR_Pred"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["target_multiplier"] + feature_cols, inplace=True)

        if len(df) <= train_window: return pd.DataFrame()
        
        preds, dates = [], []
        for start in range(train_window, len(df) - step, step):
            train_set = df.iloc[start - train_window : start]
            test_set = df.iloc[start : start + step]
            X_train, y_train = train_set[feature_cols], train_set["target_multiplier"]
            X_test, y_test = test_set[feature_cols], test_set["target_multiplier"]

            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred = model.predict(X_test)
            preds.extend(y_pred)
            dates.extend(test_set["Date"].values)

        if not preds: return pd.DataFrame()
        return pd.DataFrame({"Date": dates, "Pred_Mult": preds})

    def _rolling_forecast_garch(self, df: pd.DataFrame, future_window: int = 45, train_window: int = 500, step: int = 30) -> pd.DataFrame:
        """Performs a rolling GARCH(1,1) forecast. [REVISED & FIXED]"""
        preds, dates = [], []
        
        for start in range(train_window, len(df), step):
            train_set = df.iloc[start - train_window : start]
            test_set = df.iloc[start : start + step]
            
            if test_set.empty: continue

            # Check if the training set is large enough and has variance
            if len(train_set) < 50 or train_set['log_return'].var() < 1e-8:
                continue

            try:
                model = arch_model(train_set['log_return'], vol='Garch', p=1, q=1, dist='Normal')
                res = model.fit(disp='off', show_warning=False)
                
                # Forecast variance for 'future_window' days ahead
                forecast = res.forecast(horizon=future_window, reindex=False)
                
                # Prediction is the average of the forecasted variance over the future window
                avg_future_variance = forecast.variance.values.mean()
                
                # Apply this single forecast to all 'step' days in the current test window
                preds.extend([avg_future_variance] * len(test_set))
                dates.extend(test_set["Date"].values)

            except Exception:
                # GARCH might fail to converge. Skip this window.
                continue

        if not preds: return pd.DataFrame(columns=["Date", "GARCH_Pred"])
        return pd.DataFrame({"Date": dates, "GARCH_Pred": preds})

    def _get_random_walk_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates a naive random walk forecast where future volatility equals past volatility."""
        df_rw = df[['Date']].copy()
        # The forecast is the most recent realized volatility value (HAR's daily term)
        df_rw['RW_Pred'] = df['RV_daily'].shift(1)
        return df_rw.dropna()


    # ------------------------------------------------------------
    # Step 5: Compare Groups and Visualize
    # ------------------------------------------------------------
    def compare_groups(self):
        groups = {
            "Major_Banks": ["JPM", "BAC", "GS", "MS", "C"],
            "Sector_ETFs": ["XLK", "XLE", "XLF", "XLV", "XLY"],
            "All_Assets": ["JPM", "BAC", "GS", "MS", "C", "XLK", "XLE", "XLF", "XLV", "XLY"]
        }
        base_spy_df = pd.read_csv(self.spy_path)
        base_spy_df["Date"] = pd.to_datetime(base_spy_df["Date"])
        base_spy_df.sort_values("Date", inplace=True)

        for group_name, tickers in groups.items():
            try:
                print(f"\n{'='*25} Processing Group: {group_name} {'='*25}")
                
                asset_dfs_map, start_dates = {}, [base_spy_df['Date'].min()]
                for ticker in tickers:
                    path = f"data/{ticker}_data.csv"
                    if os.path.exists(path):
                        df = pd.read_csv(path); df.dropna(inplace=True); df["Date"] = pd.to_datetime(df["Date"])
                        if not df.empty: asset_dfs_map[ticker] = df.sort_values("Date"); start_dates.append(df['Date'].min())
                
                if not asset_dfs_map: continue

                common_start_date = max(start_dates)
                aligned_spy_df = base_spy_df[base_spy_df['Date'] >= common_start_date].copy().reset_index(drop=True)
                aligned_assets_map = {ticker: df[df['Date'] >= common_start_date].copy() for ticker, df in asset_dfs_map.items()}

                spy_with_baseline = self._prepare_spy_baseline(aligned_spy_df, future_window=45)
                features = self._build_asset_features(aligned_assets_map)
                merged = pd.merge(spy_with_baseline, features, on="Date", how="inner").dropna()

                if merged.empty: print(f"Skipping group '{group_name}' due to no overlapping data."); continue
                
                # --- Model 1: HAR-XGBoost ---
                print("--- Running Hybrid HAR-XGBoost Model ---")
                roll_df_xgb = self._rolling_forecast_xgb(merged.copy(), [c for c in features.columns if c != 'Date'])
                
                # --- Model 2: GARCH(1,1) ---
                print("--- Running GARCH(1,1) Model ---")
                roll_df_garch = self._rolling_forecast_garch(merged.copy(), future_window=45)

                # --- Model 3: Random Walk ---
                print("--- Running Random Walk Model ---")
                df_rw = self._get_random_walk_forecast(merged.copy())

                # --- Merge and Evaluate ---
                final_results = pd.merge(merged, roll_df_xgb, on="Date", how="inner")
                final_results["Hybrid_Pred"] = final_results["HAR_Pred"] * final_results["Pred_Mult"]
                final_results = pd.merge(final_results, roll_df_garch, on="Date", how="left")
                final_results = pd.merge(final_results, df_rw[["Date", "RW_Pred"]], on="Date", how="left")
                final_results.dropna(subset=['Hybrid_Pred', 'GARCH_Pred', 'RW_Pred'], inplace=True)

                if final_results.empty: print("Not enough data to create a final comparison. Skipping group."); continue

                # --- Performance Metrics ---
                models = ["Hybrid_Pred", "GARCH_Pred", "RW_Pred"]
                performance_summary = []
                for model_pred in models:
                    r2 = r2_score(final_results["RV_future"], final_results[model_pred])
                    rmse = np.sqrt(mean_squared_error(final_results["RV_future"], final_results[model_pred]))
                    performance_summary.append({"Model": model_pred.replace('_Pred', ''), "R-squared (R²)": f"{r2:.4f}", "RMSE": f"{rmse:.6f}"})

                summary_df = pd.DataFrame(performance_summary)
                print(f"\n--- Performance Summary for {group_name} ---")
                print(summary_df.to_string(index=False))

                # --- Visualization ---
                plt.style.use('seaborn-v0_8-darkgrid')
                plt.figure(figsize=(16, 8))
                plt.plot(final_results["Date"], final_results["RV_future"], label="Actual Future Volatility", color="black", alpha=0.8, linewidth=2)
                plt.plot(final_results["Date"], final_results["Hybrid_Pred"], label=f"Hybrid XGBoost (R²: {performance_summary[0]['R-squared (R²)']})", color="blue", alpha=0.8)
                plt.plot(final_results["Date"], final_results["GARCH_Pred"], label=f"GARCH (R²: {performance_summary[1]['R-squared (R²)']})", color="red", linestyle="--", alpha=0.7)
                plt.plot(final_results["Date"], final_results["RW_Pred"], label=f"Random Walk (R²: {performance_summary[2]['R-squared (R²)']})", color="green", linestyle=":", alpha=0.7)
                plt.title(f"Volatility Forecast Comparison (45-Day Horizon) | Features: {group_name}", fontsize=16)
                plt.xlabel("Date", fontsize=12); plt.ylabel("Realized Volatility", fontsize=12); plt.legend(fontsize=12); plt.tight_layout(); plt.show()

            except Exception as e:
                print(f"An unexpected error occurred while processing group {group_name}: {e}")

if __name__ == "__main__":
    forecaster = MultiAssetVolatilityForecaster(spy_path="data/SPY_data.csv", data_dir="data")
    forecaster.compare_groups()