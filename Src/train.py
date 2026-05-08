"""
=============================================================================
Milestone 2 - Online Games Popularity Prediction (Modularized)
Tasks: Regression and Classification Pipelines
=============================================================================
"""

import os
import sys

# Force working directory to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, r2_score

# Import modularized functions
from preprocessing import load_data, preprocess_data
from feature_engineering import select_features
from regression_model_training import train_and_evaluate_models as run_regression
from classification_model_training import train_and_evaluate_classification_models as run_classification

def save_df_as_table_png(df, filename, title):
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Round numerical columns for clean display
    df_display = df.round(4)
    
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title(title, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

from colors import GREEN, BLUE, ORANGE, RED, RESET, BOLD

def run_pipeline(task, train_file, target):
    print(f"\n{BLUE}{BOLD}" + "=" * 40)
    print(f"STARTING {task.upper()} PIPELINE")
    print("=" * 40 + f"{RESET}")
    
    # 1. LOAD DATA
    print(f"\n{BLUE}[1] LOADING DATA{RESET}")
    train = load_data(train_file, "Datasets/idlist.csv")
    print(f"Train shape  : {train.shape}")
    
    # 2. PREPROCESSING
    print(f"\n{BLUE}[2] PREPROCESSING{RESET}")
    df, all_features = preprocess_data(train, target=target, task=task)
    
    # 3. FEATURE SELECTION
    print(f"\n{BLUE}[3] FEATURE SELECTION{RESET}")
    X_sel, y, selected_features, scores_df = select_features(df, all_features, target=target, task=task)
    
    
    plt.figure(figsize=(10, 7))
    scores_df.head(25).set_index("Feature")["F_Score"].plot(kind="barh", color="steelblue")
    plt.title(f"Feature F-Scores ({task.capitalize()})")
    plt.xlabel("F-Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"Plots/{task}/feature_importance.png", dpi=150)
    plt.close()

    # Save tabularized feature importance
    save_df_as_table_png(
        scores_df.head(25),
        f"Tables/{task}/feature_importance_table.png",
        f"Top 25 Feature F-Scores ({task.capitalize()})"
    )

    # 4. MODEL TRAINING
    print(f"\n{BLUE}[4] MODEL TRAINING & EVALUATION{RESET}")
    if task == "regression":
        results_df, preds_test, y_test = run_regression(X_sel, y)
    else:
        results_df, preds_test, y_test = run_classification(X_sel, y)
        
    print("\nRESULTS SUMMARY — Test Set")
    print(results_df.to_string(index=False))

    # 5. PLOTS
    print(f"\n[5] SAVING {task.upper()} PLOTS & TABLES")
    
    # Save tabularized model comparison
    save_df_as_table_png(
        results_df,
        f"Tables/{task}/model_comparison_table.png",
        f"{task.capitalize()} Model Comparison"
    )

    if task == "regression":
        # Actual vs Predicted
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        for ax, (mname, ypred) in zip(axes.flatten(), preds_test.items()):
            yt = np.array(y_test)
            ax.scatter(yt, ypred, alpha=0.3, s=8, color="steelblue")
            lim = [min(yt.min(), ypred.min()), max(yt.max(), ypred.max())]
            ax.plot(lim, lim, "r--", lw=2, label="Perfect fit")
            ax.set_title(f"{mname}\nR²={r2_score(yt,ypred):.3f}  RMSE={np.sqrt(mean_squared_error(yt,ypred)):.3f}")
            ax.set_xlabel("Actual (log1p)")
            ax.set_ylabel("Predicted (log1p)")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        plt.suptitle("Actual vs Predicted — RecommendationCount", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"Plots/{task}/actual_vs_predicted.png", dpi=150)
        plt.close()

        # Model comparison bar plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
            ax.bar(results_df["Model"], results_df[metric], color=colors)
            ax.set_title(metric)
            ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.suptitle("Regression Model Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"Plots/{task}/model_comparison.png", dpi=150)
        plt.close()
        
    else:
        # Save tabularized performance time
        save_df_as_table_png(
            results_df[["Model", "Accuracy", "Train Time (s)", "Test Time (s)"]],
            f"Tables/{task}/performance_time_table.png",
            "Classification Performance Time"
        )

        # Model comparison bar plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        for ax, metric in zip(axes, ["Accuracy", "Precision", "Recall", "F1"]):
            ax.bar(results_df["Model"], results_df[metric], color=colors[:len(results_df)])
            ax.set_title(metric)
            ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.suptitle("Classification Model Comparison", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"Plots/{task}/model_comparison.png", dpi=150)
        plt.close()

        # Accuracy vs Train Time vs Test Time bar plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ["Accuracy", "Train Time (s)", "Test Time (s)"]
        for ax, metric in zip(axes, metrics):
            ax.bar(results_df["Model"], results_df[metric], color=colors[:len(results_df)])
            ax.set_title(metric)
            ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.suptitle("Classification: Accuracy vs Training Time vs Testing Time", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"Plots/{task}/performance_time.png", dpi=150)
        plt.close()

    print(f"\n{GREEN}[6] {task.upper()} PIPELINE COMPLETED{RESET}")

def main():
    os.makedirs("Plots/Regression", exist_ok=True)
    os.makedirs("Plots/Classification", exist_ok=True)
    os.makedirs("Tables/Regression", exist_ok=True)
    os.makedirs("Tables/Classification", exist_ok=True)
    
    # Run Regression Pipeline
    run_pipeline(
        task="regression",
        train_file="Datasets/regression_train_data.csv",
        target="RecommendationCount"
    )
    
    # Run Classification Pipeline
    run_pipeline(
        task="classification",
        train_file="Datasets/classification_train_data.csv",
        target="GamePopularity"
    )

    print("\n" + "=" * 60)
    print("MILESTONE 2 COMPLETE ✓")
    print("All plots and tabular PNGs saved in 'Plots' directory")
    print("=" * 60)

if __name__ == "__main__":
    main()