import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class VoloModel:
    def __init__(self, window=45):
        self.window = window

    def rolling_xgb_pca_fold(
        self,
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        train_window=100,
        step=5,
        fold_interval=5,
        low_importance_ratio=0.3,
        n_pca_components=3
    ):

        if "Date" not in x_df.columns:
            if x_df.index.name == "Date":
                x_df = x_df.reset_index()
            else:
                raise ValueError("x_df must contain 'Date' column or have 'Date' as index name.")
        if "Date" not in y_df.columns:
            if y_df.index.name == "Date":
                y_df = y_df.reset_index()
            else:
                raise ValueError("y_df must contain 'Date' column or have 'Date' as index name.")

        merged = pd.merge(x_df, y_df, on="Date", how="inner")
        target_col = [c for c in y_df.columns if c != "Date"][0]
        x_cols = [c for c in x_df.columns if c != "Date"]

        preds, actuals, dates = [], [], []
        all_feature_importance = []
        global_importance = pd.Series(0.0, index=x_cols)
        scaler = StandardScaler()

        try:
            _ = xgb.DeviceQuantileDMatrix
            use_gpu = True
        except Exception:
            use_gpu = False

        for i, start in enumerate(range(train_window, len(merged) - step, step)):
            train_set = merged.iloc[start - train_window:start]
            test_set = merged.iloc[start:start + step]

            X_train = train_set[x_cols]
            y_train = train_set[target_col]
            X_test = test_set[x_cols]
            y_test = test_set[target_col]

            params = dict(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                # Use GPU if available
                device="gpu" if use_gpu else "cpu",
                tree_method="hist" if use_gpu else "hist"
            )
            

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            imp = model.get_booster().get_score(importance_type="gain")
            imp_series = pd.Series(imp, dtype=float).reindex(x_cols, fill_value=0)
            global_importance = 0.9 * global_importance.add(imp_series, fill_value=0)

            all_feature_importance.append(imp_series.to_dict())
            y_pred = model.predict(X_test)

            # Constrain predictions within training y range with 5% buffer
            min_train_y = y_train.min()
            max_train_y = y_train.max()
            lower_bound = min_train_y - 0.05 * abs(min_train_y)
            upper_bound = max_train_y + 0.05 * abs(max_train_y)
            y_pred_constrained = np.clip(y_pred, lower_bound, upper_bound)

            preds.extend(y_pred_constrained)
            actuals.extend(y_test.values)
            dates.extend(test_set["Date"].values)

            if (i + 1) % fold_interval == 0 and len(x_cols) > 10:
                avg_imp = global_importance.copy()
                avg_imp /= avg_imp.sum()
                sorted_features = avg_imp.sort_values(ascending=True)
                low_imp_features = sorted_features.head(int(len(sorted_features) * low_importance_ratio)).index.tolist()

                if len(low_imp_features) >= n_pca_components:
                    print(f"\n[PCA Fold] Window {start}: Combining {len(low_imp_features)} low-importance features into {n_pca_components} PCA components")

                    pca_input = merged[low_imp_features].fillna(0).values
                    scaled_input = scaler.fit_transform(pca_input)
                    pca = PCA(n_components=n_pca_components, random_state=42)
                    pca_features = pca.fit_transform(scaled_input)

                    for j in range(n_pca_components):
                        merged[f"PCA_Fold_{i+1}_{j+1}"] = pca_features[:, j]

                    x_cols = [c for c in x_cols if c not in low_imp_features] + [f"PCA_Fold_{i+1}_{j+1}" for j in range(n_pca_components)]
                    global_importance = global_importance.loc[[f for f in global_importance.index if f in x_cols]]
                    for new_f in [f for f in x_cols if f.startswith(f"PCA_Fold_{i+1}")]:
                        global_importance[new_f] = 0.1

        
        
        result_df = pd.DataFrame({
            "Date": pd.to_datetime(dates),
            "Actual": actuals,
            "Predicted": preds
        }).sort_values("Date").set_index("Date")

        final_importance = global_importance.sort_values(ascending=False).reset_index()
        final_importance.columns = ["Feature", "Avg_Gain"]
        final_importance.to_csv("results/feature_pca_folding_importance.csv", index=False)

        print("\n=== Final Feature Importance (Averaged) ===")
        print(final_importance.head(15).to_string(index=False))
        print(f"\nRemaining Features: {len(x_cols)}")

        return result_df, final_importance
