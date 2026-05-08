import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

def select_features(df, features, target="RecommendationCount", task="regression", k_max=25, is_train=True):
    X = df[features].copy()
    
    from save_model import save_object, load_object
    
    if is_train:
        y = df[target].copy()
        K = min(k_max, X.shape[1])
        score_func = f_regression if task == "regression" else f_classif
            
        selector = SelectKBest(score_func=score_func, k=K)
        selector.fit(X, y)
        save_object(selector, f"Models/{task}/Selector.pkl")
        
        scores_df = pd.DataFrame({"Feature": X.columns, "F_Score": selector.scores_}) \
                      .sort_values("F_Score", ascending=False)
    else:
        selector = load_object(f"Models/{task}/Selector.pkl")
        y = None
        scores_df = None
        
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"  Selected top {len(selected_features)} features.")
                  
    return X[selected_features], y, selected_features, scores_df
