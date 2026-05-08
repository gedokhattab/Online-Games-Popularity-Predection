import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from save_model import save_object
from colors import GREEN, ORANGE, RED, RESET

def evaluate(name, y_true, y_pred, subset="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"    [{subset}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return mae, rmse, r2

def train_and_evaluate_models(X, y):
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.15 / 0.85, random_state=42)

    print(f"  Training set   : {X_train.shape[0]:,} rows  (~70%)")
    print(f"  Validation set : {X_val.shape[0]:,} rows  (~15%)")
    print(f"  Test set       : {X_test.shape[0]:,} rows  (~15%)")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    
    save_object(scaler, "Models/regression/Scaler.pkl")

    results = []
    preds_test = {}

    param_grids = {
        "Ridge Regression": {
            "alpha": [0.1, 1.0, 10.0]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
            "min_samples_leaf": [1, 2, 5]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        }
    }

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- Tuning & Training: {name} ---")
        
        # Use GridSearchCV for models with defined parameters
        if name in param_grids:
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grids[name], 
                cv=3, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            if name == "Ridge Regression":
                grid_search.fit(X_train_sc, y_train)
            else:
                grid_search.fit(X_train, y_train)
                
            model = grid_search.best_estimator_
            print(f"    Best Params: {grid_search.best_params_}")
        else:
            # Linear Regression (no tuning needed)
            model.fit(X_train_sc, y_train)

        # Evaluation
        if name in ["Linear Regression", "Ridge Regression"]:
            evaluate(name, y_val, model.predict(X_val_sc), "Validation")
            mae, rmse, r2 = evaluate(name, y_test, model.predict(X_test_sc), "Test")
            preds_test[name] = model.predict(X_test_sc)
        else:
            evaluate(name, y_val, model.predict(X_val), "Validation")
            mae, rmse, r2 = evaluate(name, y_test, model.predict(X_test), "Test")
            preds_test[name] = model.predict(X_test)
        
        save_object(model, f"Models/regression/{name.replace(' ', '_')}.pkl")
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    results_df = pd.DataFrame(results)
    
    return results_df, preds_test, y_test
