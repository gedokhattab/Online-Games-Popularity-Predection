import os
import sys

# Force working directory to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Src"))

import pandas as pd
import numpy as np

# Import our modularized functions
from preprocessing import load_data, preprocess_data
from feature_engineering import select_features
from save_model import load_object

def test_pipeline(test_file=None, task=None):
    print("=" * 60)
    print("MILESTONE 2: TEST SCRIPT")
    print("=" * 60)
    
    if not test_file:
        test_file = input("\nEnter the absolute path to testfile.csv: ").strip()
        
    if not os.path.exists(test_file):
        print(f"Error: Could not find file at {test_file}")
        return

    if not task:
        print("\nSelect the task type:")
        print("  1. Regression")
        print("  2. Classification")
        task_choice = input("Enter 1 or 2: ").strip()
        
        if task_choice == "1":
            task = "regression"
        elif task_choice == "2":
            task = "classification"
        else:
            print("Error: Invalid choice. Must be 1 or 2.")
            return

    target_col = "RecommendationCount" if task == "regression" else "GamePopularity"
    
    print("\n[1] LOADING TEST DATA")
    try:
        df = load_data(test_file, "Datasets/idlist.csv")
        print(f"Loaded test data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n[2] PREPROCESSING TEST DATA (Using saved imputer)")
    # 'is_train=False' will load the saved imputer instead of fitting a new one
    try:
        df_preprocessed, all_features = preprocess_data(df, target=target_col, task=task, is_train=False)
        print("Preprocessing complete.")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return

    print("\n[3] FEATURE SELECTION (Using saved selector)")
    # 'is_train=False' will load the saved selector instead of fitting a new one
    try:
        X_sel, _, selected_features, _ = select_features(
            df_preprocessed, all_features, target=target_col, task=task, is_train=False
        )
        print(f"Applied saved feature selector. Selected {len(selected_features)} features.")
    except Exception as e:
        print(f"Error in feature selection: {e}")
        return

    print("\n[4] LOADING SCALER AND MODELS")
    try:
        scaler = load_object(f"Models/{task}/Scaler.pkl")
        
        # Determine which models to run based on the task
        if task == "regression":
            model_names = ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]
            # Models that require scaled data
            needs_scaling = ["Linear Regression", "Ridge Regression"]
        else:
            model_names = ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"]
            needs_scaling = ["Logistic Regression"]
            
        X_scaled = scaler.transform(X_sel)
        
        predictions = {}
        for m_name in model_names:
            filename = f"Models/{task}/{m_name.replace(' ', '_')}.pkl"
            if os.path.exists(filename):
                model = load_object(filename)
                
                # Predict
                if m_name in needs_scaling:
                    preds = model.predict(X_scaled)
                else:
                    preds = model.predict(X_sel)
                
                # If regression, we need to convert back from log1p space using expm1
                if task == "regression":
                    preds = np.expm1(preds)
                
                predictions[m_name] = preds
                print(f"  -> Successfully generated predictions using {m_name}")
            else:
                print(f"  -> Skipping {m_name} (Model file not found)")
                
    except Exception as e:
        print(f"Error loading models or predicting: {e}")
        return

    print("\n[5] SAVING PREDICTIONS")
    # Save predictions to a CSV
    output_df = pd.DataFrame()
    # If the original dataframe had a QueryID or ID, we can use it
    if "QueryID" in df.columns:
        output_df["QueryID"] = df["QueryID"]
        
    for m_name, preds in predictions.items():
        output_df[f"Predicted_{m_name.replace(' ', '_')}"] = preds

    output_filename = f"{task}_predictions_output.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Predictions successfully saved to: {output_filename}")
    
    print("\nSample of predictions:")
    print(output_df.head())
    print("\n" + "=" * 60)
    print("TEST PIPELINE COMPLETE ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
