import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from save_model import save_object
from colors import GREEN, ORANGE, RED, RESET

def evaluate_classification(name, y_true, y_pred, subset="Test"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"    [{subset}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}{RESET}")
    return acc, prec, rec, f1

def train_and_evaluate_classification_models(X, y):
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.15 / 0.85, random_state=42)

    print(f"  Training set   : {X_train.shape[0]:,} rows  (~70%){RESET}")
    print(f"  Validation set : {X_val.shape[0]:,} rows  (~15%){RESET}")
    print(f"  Test set       : {X_test.shape[0]:,} rows  (~15%){RESET}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    
    save_object(scaler, "Models/classification/Scaler.pkl")

    results = []
    preds_test = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- Model: {name} ---")
        train_start = time.time()
        
        if name in ["Logistic Regression"]:
            model.fit(X_train_sc, y_train)
            train_end = time.time()
            evaluate_classification(name, y_val, model.predict(X_val_sc), "Validation")
            
            test_start = time.time()
            y_pred_test = model.predict(X_test_sc)
            test_end = time.time()
            
            acc, prec, rec, f1 = evaluate_classification(name, y_test, y_pred_test, "Test")
            preds_test[name] = y_pred_test
        else:
            model.fit(X_train, y_train)
            train_end = time.time()
            evaluate_classification(name, y_val, model.predict(X_val), "Validation")
            
            test_start = time.time()
            y_pred_test = model.predict(X_test)
            test_end = time.time()
            
            acc, prec, rec, f1 = evaluate_classification(name, y_test, y_pred_test, "Test")
            preds_test[name] = y_pred_test
        
        train_time = train_end - train_start
        test_time = test_end - test_start
        print(f"    [Time] Train={train_time:.3f}s, Test={test_time:.3f}s")
        
        save_object(model, f"Models/classification/{name.replace(' ', '_')}.pkl")
        
        results.append({
            "Model": name, 
            "Accuracy": acc, 
            "Precision": prec, 
            "Recall": rec, 
            "F1": f1,
            "Train Time (s)": train_time,
            "Test Time (s)": test_time
        })

    results_df = pd.DataFrame(results)
    
    return results_df, preds_test, y_test
