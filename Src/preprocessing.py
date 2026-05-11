import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(train_path="Datasets/regression_train_data.csv", idlist_path="Datasets/idlist.csv"):
    from sklearn.model_selection import train_test_split
    train = pd.read_csv(train_path)
    idlist = pd.read_csv(idlist_path)
    idlist.columns = idlist.columns.str.strip()
    train = train.merge(idlist, left_on="QueryID", right_on="ID", how="left")
    
    train_df, test_df = train_test_split(train, test_size=0.15, random_state=42)
    return train_df, test_df

def preprocess_data(df, target="RecommendationCount", task="regression", is_train=True):
    # 2a. Drop text
    drop_text = [
        "QueryID", "ResponseID", "QueryName", "ResponseName",
        "ID", "GameName", "SupportEmail", "SupportURL", "AboutText", "Background",
        "ShortDescrip", "DetailedDescrip", "DRMNotice", "ExtUserAcctNotice", 
        "HeaderImage", "LegalNotice", "Reviews", "SupportedLanguages", "Website",
        "PCMinReqsText", "PCRecReqsText", "LinuxMinReqsText", "LinuxRecReqsText",
        "MacMinReqsText", "MacRecReqsText", "PriceCurrency"
    ]
    dropped = [c for c in drop_text if c in df.columns]
    df.drop(columns=dropped, inplace=True)

    # 2b. Parse ReleaseDate
    if "ReleaseDate" in df.columns:
        parsed = pd.to_datetime(df["ReleaseDate"], errors="coerce")
        df["ReleaseYear"]  = parsed.dt.year
        df["ReleaseMonth"] = parsed.dt.month
        df.drop(columns=["ReleaseDate"], inplace=True)

    # 2c & 2d. Boolean cols
    bool_cols = [
        "ControllerSupport", "IsFree", "FreeVerAvail", "PurchaseAvail",
        "SubscriptionAvail", "PlatformWindows", "PlatformLinux", "PlatformMac",
        "PCReqsHaveMin", "PCReqsHaveRec", "LinuxReqsHaveMin", "LinuxReqsHaveRec",
        "MacReqsHaveMin", "MacReqsHaveRec", "CategorySinglePlayer",
        "CategoryMultiplayer", "CategoryCoop", "CategoryMMO",
        "CategoryInAppPurchase", "CategoryIncludeSrcSDK",
        "CategoryIncludeLevelEditor", "CategoryVRSupport",
        "GenreIsNonGame", "GenreIsIndie", "GenreIsAction", "GenreIsAdventure",
        "GenreIsCasual", "GenreIsStrategy", "GenreIsRPG", "GenreIsSimulation",
        "GenreIsEarlyAccess", "GenreIsFreeToPlay", "GenreIsSports",
        "GenreIsRacing", "GenreIsMassivelyMultiplayer",
    ]
    bool_cols = [c for c in bool_cols if c in df.columns]
    for col in bool_cols:
        df[col] = df[col].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # 2e. Numeric cols
    numeric_cols = [
        "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
        "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
        "SteamSpyOwners", "SteamSpyOwnersVariance",
        "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
        "AchievementCount", "AchievementHighlightedCount",
        "PriceInitial", "PriceFinal", "ReleaseYear", "ReleaseMonth",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    from save_model import save_object, load_object
    if is_train:
        num_imputer = SimpleImputer(strategy="median")
        if numeric_cols:
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        print(f"  Shape after preprocessing : {df.shape}")
        save_object(num_imputer, f"Models/{task}/Imputer.pkl")
    else:
        num_imputer = load_object(f"Models/{task}/Imputer.pkl")
        if numeric_cols:
            df[numeric_cols] = num_imputer.transform(df[numeric_cols])

    # 2f. Target imputation
    if target in df.columns and df[target].isnull().sum() > 0:
        if task == "regression":
            df[target] = df[target].fillna(df[target].median())
        else:
            df[target] = df[target].fillna(df[target].mode()[0])

    # 2g. IQR cap remove outliers
    for col in numeric_cols:
        if col == target: continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = df[col].clip(lo, hi)

    # 2h. Log1p transform normalize skewed data
    skew_cols = [
        "SteamSpyOwners", "SteamSpyOwnersVariance",
        "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance"
    ]
    if task == "regression":
        skew_cols.append(target)
        
    skew_cols = [c for c in skew_cols if c in df.columns]
    for col in skew_cols:
        df[col] = np.log1p(df[col].clip(lower=0))

    all_features = [c for c in bool_cols + numeric_cols if c in df.columns and c != target]

    return df, all_features
