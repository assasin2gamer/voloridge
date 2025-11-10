import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
import os
import glob
import matplotlib.pyplot as plt

class VoloFeatureImplV2:
    def __init__(self, window=45):
        self.window = window
        
        
    def feature_process_var_index(self, index_name, future_window):
        index_data = pd.read_csv(f"data/{index_name}_data.csv", parse_dates=["Date"])
        index_data.set_index("Date", inplace=True)

        # Prepare future realized variance and HAR model for index
        index_data["RV_future"] = (
            index_data["Log_Return"]
            .shift(-future_window)                # look forward
            .rolling(window=future_window)        # take variance over the *future* window
            .var()
        )        
        y = index_data["RV_future"]
        index_data['RQ_daily'] = index_data['Log_Return']**4
        index_data['RV_x_RQ'] = index_data['Variance_45d'] * np.sqrt(index_data['RQ_daily'])
        X = add_constant(index_data[["Variance_45d", "Variance_45d_MA", "Variance_5d_MA"]])
        har_model = OLS(y, X, missing="drop").fit()
        index_data["HAR_Pred"] = har_model.predict(X)
        index_data["target_variance_index_multiplier"] = index_data["RV_future"] / index_data["HAR_Pred"]
        index_data.dropna(inplace=True)
        # Make sure it is a pd dataframe
        index_data = index_data[["HAR_Pred", "target_variance_index_multiplier"]]
        
        #print(index_data.head())
        return index_data
    
    def feature_process_var_corr(self, index_name, stock_name, future_window):
        index_data = pd.read_csv(f"data/{index_name}_data.csv", parse_dates=["Date"])
        index_data.set_index("Date", inplace=True)

        stock_data = pd.read_csv(f"data/{stock_name}_data.csv", parse_dates=["Date"])
        stock_data.set_index("Date", inplace=True)
        combined_data = index_data.join(stock_data, how="inner", lsuffix=f"_{index_name}", rsuffix=f"_{stock_name}")

        # Rolling covariance
        combined_data[f"Cov_45_{stock_name}"] = (
            combined_data[f"Log_Return_{index_name}"].rolling(window=future_window).cov(combined_data[f"Log_Return_{stock_name}"])
        )
        
        # Rolling variance of stock
        combined_data[f"Var_45_{stock_name}"] = (
            combined_data[f"Log_Return_{stock_name}"].rolling(window=future_window).var()
        )
        combined_data[f"Var_45_{stock_name}_MA"] = (
            combined_data[f"Var_45_{stock_name}"].rolling(window=future_window).mean()
        )
        combined_data[f"Var_5_{stock_name}_MA"] = (
            combined_data[f"Var_45_{stock_name}"].rolling(window=5).mean()
        )
        # var _qsquared
        combined_data[f"Var_Sq_{stock_name}"] = combined_data[f"Var_45_{stock_name}"] ** 2
        combined_data[f"Var_Log_{stock_name}"] = np.log1p(combined_data[f"Var_45_{stock_name}"])
        
        
        # Rolling correlation
        combined_data[f"Corr_45_{stock_name}"] = (
            combined_data[f"Log_Return_{index_name}"].rolling(window=future_window).corr(combined_data[f"Log_Return_{stock_name}"])
        )
        combined_data[f"Corr_45_{stock_name}_MA"] = (
            combined_data[f"Corr_45_{stock_name}"].rolling(window=future_window).mean()
        )
        combined_data[f"Corr_5_{stock_name}"] = (
        combined_data[f"Corr_45_{stock_name}"].rolling(5).mean()
        )
        combined_data[f"Corr_22_{stock_name}"] = (
            combined_data[f"Corr_45_{stock_name}"].rolling(22).mean()
        )
        combined_data[f"Corr_Sq_{stock_name}"] = combined_data[f"Corr_45_{stock_name}"] ** 2
        combined_data[f"Corr_Log_{stock_name}"] = np.log1p(combined_data[f"Corr_45_{stock_name}"])
        combined_data[f"Var_Sum_{stock_name}"] = (
            combined_data["Variance_45d_SPY"] + combined_data[f"Variance_45d_{stock_name}"]
        )
        combined_data[f"Cov_Norm_{stock_name}"] = (
            combined_data[f"Cov_45_{stock_name}"] / combined_data[f"Var_Sum_{stock_name}"]
        )

        # Forward stock covariance
        combined_data[f"Future_Cov_45_{stock_name}"] = (
            combined_data[f"Log_Return_{stock_name}"].shift(-future_window)
            .rolling(window=future_window)
            .cov(combined_data[f"Log_Return_{index_name}"].shift(-future_window))
        )

        # Forward stock variance
        combined_data[f"Future_Var_45_{stock_name}"] = (
            combined_data[f"Log_Return_{stock_name}"].shift(-future_window)
            .rolling(window=future_window)
            .var()
        )

        # Forward correlation (target)
        combined_data[f"Future_Corr_45_{stock_name}"] = (
            combined_data[f"Log_Return_{index_name}"].shift(-future_window)
            .rolling(window=future_window)
            .corr(combined_data[f"Log_Return_{stock_name}"].shift(-future_window))
        )
        
        # Create HAR-style regression features for both correlation and variance
        # Fit OLS HAR-style regression on future correlation
        y_corr = combined_data[f"Future_Corr_45_{stock_name}"]
        X_corr = add_constant(
            combined_data[
            [
                f"Corr_45_{stock_name}",    # daily/short-term component
                f"Corr_5_{stock_name}",     # weekly
                f"Corr_22_{stock_name}",    # monthly
                f"Corr_45_{stock_name}_MA", # long-term average
                f"Var_Sum_{stock_name}",
                f"Cov_Norm_{stock_name}",
                f"Corr_Sq_{stock_name}",
                f"Corr_Log_{stock_name}",
            ]
        ])
        
        model_corr = OLS(y_corr, X_corr, missing="drop").fit()
        # Add prediction and residuals
        combined_data[f"HAR_Pred_Corr_{stock_name}"] = model_corr.predict(X_corr)
        # Fit OLS HAR-style regression on stock variance
        y_var = combined_data[f"Log_Return_{stock_name}"].shift(-future_window).rolling(window=future_window).var()
        X_var = add_constant(combined_data[
            [
                f"Var_45_{stock_name}",
                f"Var_45_{stock_name}_MA",
                f"Var_5_{stock_name}_MA",
                f"Var_Sq_{stock_name}",
                f"Var_Log_{stock_name}",
                f"Var_5_{stock_name}_MA"
                
            ]
        ])
        model_var = OLS(y_var, X_var, missing="drop").fit()
        combined_data[f"HAR_Pred_Var_{stock_name}"] = model_var.predict(X_var)
        
        
        print(combined_data.head())
        combined_data["target_correlation"] = combined_data[f"Future_Corr_45_{stock_name}"]/combined_data[f"HAR_Pred_Corr_{stock_name}"]
        combined_data["target_variance"] = combined_data[f"Future_Var_45_{stock_name}"]/combined_data[f"HAR_Pred_Var_{stock_name}"]
        combined_data.dropna(inplace=True)
       
        
        return combined_data
    
    def feature_collector(self, stock_names):
        # For each stock_names, from data/{stock_name}_data.csv, merge
        all_features = []
        for stock_name in stock_names:
            stock_data = pd.read_csv(f"data/{stock_name}_data.csv", parse_dates=["Date"])
            stock_data.set_index("Date", inplace=True)
            features = stock_data[[f"Log_Return", f"Beta_45d", f"Variance_45d", f"Variance_45d_MA", f"Variance_5d_MA", f"MA_45d", f"MA_5d"]]
            # Rename columns to include stock name
            features.columns = [f"{col}_{stock_name}" for col in features.columns]
            all_features.append(features)
        combined_features = pd.concat(all_features, axis=1)
        combined_features.reset_index(inplace=True)
        # remove anything with any text with future in the column name
        combined_features = combined_features.loc[:, 
            ~combined_features.columns.str.contains('Future', case=False) &
            ~combined_features.columns.str.contains('target', case=False)
        ]
        # remove any column not numeric
        combined_features = combined_features.loc[:, ~combined_features.columns.str.contains('Future_Beta_45d')]
        combined_features = combined_features.loc[:, ~combined_features.columns.str.contains('Code')]
        combined_features = combined_features.loc[:, ~combined_features.columns.str.contains('RV_future')]
        combined_features = combined_features.loc[:, ~combined_features.columns.str.contains('Symbol')]
        combined_features = combined_features.loc[:, ~combined_features.columns.str.contains('Residual')]
        combined_features.dropna(inplace=True)
        # set index to Date
        combined_features.set_index("Date", inplace=True)
        
        return combined_features
        