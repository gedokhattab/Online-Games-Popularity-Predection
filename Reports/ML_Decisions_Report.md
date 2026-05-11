# 🎮 Online Games Popularity Prediction — ML Decisions Report

> **Audience:** ML student learning *why* each pipeline decision was made, not just *what* was done.  
> **Dataset:** 78 columns × ~N rows. Targets: `RecommendationCount` (regression) / `GamePopularity` (classification).

---

## Table of Contents

1. [Preprocessing](#1-preprocessing)
   - 1.1 [Why Drop Text Columns?](#11-why-drop-text-columns)
   - 1.2 [Why No Categorical Encoding?](#12-why-no-categorical-encoding)
   - 1.3 [How Numeric Values Were Handled](#13-how-numeric-values-were-handled)
   - 1.4 [What Is Target Imputation?](#14-what-is-target-imputation)
   - 1.5 [How We Deal With Outliers](#15-how-we-deal-with-outliers)
   - 1.6 [Why Apply Log Transformation?](#16-why-apply-log-transformation)
   - 1.7 [Why StandardScaler After Transformation? (And Why Not for Trees)](#17-why-standardscaler-after-transformation-and-why-not-for-trees)
2. [Feature Engineering](#2-feature-engineering)
   - 2.1 [How We Measure the Relation Between Feature and Target](#21-how-we-measure-the-relation-between-feature-and-target)
   - 2.2 [Why Select at Most 25 Features?](#22-why-select-at-most-25-features)
3. [Regression Models](#3-regression-models)
   - 3.1 [Why These 4 Models?](#31-why-these-4-models)
   - 3.2 [Why These Hyperparameter Values?](#32-regression-hyperparameter-grids--rationale)
   - 3.3 [How Hyperparameters Affect Prediction](#33-how-hyperparameters-affect-regression-prediction)
4. [Classification Models](#4-classification-models)
   - 4.1 [Why These 3 Models?](#41-why-these-3-models)
   - 4.2 [Why These Hyperparameter Values?](#42-classification-hyperparameter-grids--rationale)
   - 4.3 [How Hyperparameters Affect Prediction](#43-how-hyperparameters-affect-classification-prediction)

---

## 1. Preprocessing

### 1.1 Why Drop Text Columns?

The pipeline drops **24 text-heavy columns** including `GameName`, `AboutText`, `DetailedDescrip`, `SupportedLanguages`, `Reviews`, `HeaderImage`, `Website`, `PCMinReqsText`, etc.

**The core reason: these columns are unstructured free-text, and classical ML models cannot consume raw text.**

| Column Type | Example | Why Dropped |
|---|---|---|
| Identifiers | `QueryID`, `ResponseID`, `QueryName` | Unique per row → zero predictive signal, would cause overfitting |
| URL / Image paths | `HeaderImage`, `SupportURL`, `Website` | Strings with no numeric meaning |
| Free-form descriptions | `AboutText`, `DetailedDescrip`, `ShortDescrip` | Would require NLP (TF-IDF, BERT) — out of scope |
| Legal/admin text | `DRMNotice`, `LegalNotice`, `ExtUserAcctNotice` | No relationship to popularity |
| System requirements text | `PCMinReqsText`, `MacRecReqsText`, etc. | Semi-structured prose — no direct numeric signal |
| Currency string | `PriceCurrency` | The actual numeric values (`PriceInitial`, `PriceFinal`) are kept instead |

> **What if we kept them?** A model like Linear Regression or Random Forest would crash on strings. To use them properly, you'd need NLP pipelines (TF-IDF vectorization, word embeddings, or transformers like BERT). That turns this into a multi-modal problem — valid research, but far beyond a standard ML course milestone.

---

### 1.2 Why No Categorical Encoding?

A common student question: *"Shouldn't we One-Hot Encode things like genre or category?"*

**The answer: it was already done — at the data creation stage, not the preprocessing stage.**

Look at the dataset columns:

```
GenreIsIndie        → bool (True/False)
GenreIsAction       → bool (True/False)
GenreIsRPG          → bool (True/False)
CategoryMultiplayer → bool (True/False)
PlatformWindows     → bool (True/False)
```

Every categorical property (Genre, Category, Platform) has **already been exploded into binary indicator columns** (a.k.a. dummy variables / one-hot encoding). The original multi-value string `"Action,Indie,RPG"` was pre-processed into three separate `1/0` columns.

The preprocessing code maps these booleans from Python `True/False` or string `"TRUE"/"FALSE"` to integer `1/0`:

```python
df[col] = df[col].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, ...}).fillna(0).astype(int)
```

> **Why not re-encode `SupportedLanguages`?** That column contains comma-separated lists like `"English, French, German"`. A proper encoding would require **Multi-Label Binarization** (one column per language). While theoretically valid, it would add hundreds of sparse columns, most with near-zero variance, likely introducing noise rather than signal. The column was dropped for simplicity and stability.

---

### 1.3 How Numeric Values Were Handled

**Step 1 — Identify numeric columns explicitly:**

```python
numeric_cols = [
    "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
    "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
    "SteamSpyOwners", "SteamSpyOwnersVariance",
    "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
    "AchievementCount", "AchievementHighlightedCount",
    "PriceInitial", "PriceFinal", "ReleaseYear", "ReleaseMonth"
]
```

**Step 2 — Handle missing values with Median Imputation (`SimpleImputer`):**

```python
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
```

**Why median, not mean?**

| Strategy | When to Use | Problem |
|---|---|---|
| **Mean** | Data is symmetric (normal distribution) | Sensitive to outliers — one game with 10M owners drags the mean up |
| **Median** | Data is skewed (which ours is) | Robust — the middle value is unaffected by extremes |
| **Mode** | Categorical / discrete | Wrong for continuous counts |

Steam game data is **heavily right-skewed**: most games have near-zero owners, a tiny few have millions. Median imputation is the correct choice here.

**Step 3 — The imputer is persisted:**

```python
save_object(num_imputer, f"Models/{task}/Imputer.pkl")
```

This means when running on **test data**, we reuse the *same* median values learned from training — we never "peek" at test data statistics. This prevents **data leakage**.

**Step 4 — ReleaseDate is parsed into numeric features:**

```python
parsed = pd.to_datetime(df["ReleaseDate"], errors="coerce")
df["ReleaseYear"]  = parsed.dt.year
df["ReleaseMonth"] = parsed.dt.month
```

Raw date strings like `"Nov 10, 2017"` cannot be fed to a model. Extracting year and month gives the model meaningful numeric signals (e.g., newer games trend differently, seasonal release patterns).

---

### 1.4 What Is Target Imputation?

The **target column** (`RecommendationCount` or `GamePopularity`) may itself have missing values. Dropping those rows would discard training data. Instead, we fill them **using different strategies per task**:

```python
if task == "regression":
    df[target] = df[target].fillna(df[target].median())
else:
    df[target] = df[target].fillna(df[target].mode()[0])
```

| Task | Strategy | Why |
|---|---|---|
| **Regression** | Fill with **median** | Target is continuous/numeric — median is robust to skew |
| **Classification** | Fill with **mode** (most frequent class) | Target is categorical — the most common class is the safest neutral guess |

> **Why not drop rows with missing targets?** With large datasets, every row is valuable. If only 1–2% of targets are missing, imputation preserves those rows with a reasonable estimate rather than discarding information.

> **Important:** Target imputation is applied **only to training data**. In production/testing, you never have the target — that's what you're predicting.

---

### 1.5 How We Deal With Outliers

**Method: IQR Clipping (Winsorization)**

```python
Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
IQR = Q3 - Q1
lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df[col] = df[col].clip(lo, hi)
```

**How it works visually:**

```
|-- 1.5×IQR ---|--- Q1 ---|--- Median ---|--- Q3 ---|--- 1.5×IQR --|
       ^                                                    ^
   lo bound                                            hi bound

Any value below lo → replaced with lo
Any value above hi → replaced with hi
```

**Why clipping instead of removing?**

| Approach | Effect | Problem |
|---|---|---|
| **Remove outliers** | Cleaner distribution | Loses rows permanently; may remove valid extremes |
| **Clip (Winsorize)** | Preserves row count | Brings extreme values to boundary — model still sees them, but not distorted |
| **Keep outliers** | No data loss | Linear models are heavily distorted; trees are less affected but still noisy |

This is applied to **all numeric input features** — but **not** the target column. Outliers in inputs confuse the model. The target's natural range is important to preserve (after log-transform).

---

### 1.6 Why Apply Log Transformation?

**Columns transformed:**

```python
skew_cols = [
    "SteamSpyOwners", "SteamSpyOwnersVariance",
    "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance"
]
# Plus RecommendationCount for the regression target
```

**Formula used:** `log1p(x)` = `log(x + 1)` — the `+1` handles zero values safely.

**Why does skew matter?**

Steam ownership data looks like this:

```
Game A:           500 owners
Game B:         5,000 owners
Game C:       500,000 owners
Game D:    50,000,000 owners  ← Minecraft, CS:GO, etc.
```

On a linear scale, Game D dominates everything. Linear regression tries to minimize squared error — a single massive value warps the entire fit.

After `log1p`:

```
log1p(500)         ≈  6.2
log1p(5,000)       ≈  8.5
log1p(500,000)     ≈ 13.1
log1p(50,000,000)  ≈ 17.7
```

The values are now **proportional and compressed** — the model can learn patterns across the whole range.

> **Why not apply to all numeric columns?** Log transform is only beneficial for heavily right-skewed distributions. Columns like `RequiredAge` (0–18), `DemoCount` (0–5), or `MovieCount` (0–50) are already on reasonable scales. Applying `log1p` to nearly-normal data actually *hurts* model performance by distorting a well-behaved distribution.

---

### 1.7 Why StandardScaler After Transformation? (And Why Not for Trees)

**What StandardScaler does:**

```
scaled_x = (x - mean) / std
```

It shifts every feature to have **mean = 0** and **standard deviation = 1**. It does **not** change the shape of the distribution — only its position and spread.

**Why is it needed after log-transform?**

Log-transform fixes *skewness* (shape). StandardScaler fixes *scale* (magnitude). They solve completely different problems:

```
After log1p + clipping:
  SteamSpyOwners      → values in [0, 18]
  ReleaseYear         → values in [1998, 2023]
  RequiredAge         → values in [0, 18]
  PriceInitial        → values in [0, 60]
```

These features now have good shapes but live at very different magnitudes. Linear Regression computes:

```
prediction = w₁×SteamSpyOwners + w₂×ReleaseYear + w₃×RequiredAge + ...
```

Without scaling, the optimizer is forced to shrink `w₂` artificially (because ReleaseYear ≈ 2015) while inflating `w₃` (because RequiredAge ≈ 7) just to compensate for magnitude differences. The model wastes its capacity fighting unit differences instead of learning real patterns.

After StandardScaler, all features live on the same ±1–2 range → coefficients are **directly comparable** and the optimizer converges faster and more correctly.

**The full preprocessing pipeline in order:**

```
Raw data
   ↓
[Clip outliers]     → fixes extreme tail values
   ↓
[log1p transform]   → fixes skewness (shape of distribution)
   ↓
[StandardScaler]    → fixes scale (magnitude between features)
```

Each step solves a different problem. Skipping any one leaves a different kind of distortion.

---

**Why StandardScaler is NOT applied to Random Forest and Gradient Boosting:**

Tree models make decisions by splitting on thresholds:

```
if ReleaseYear > 2015 → go left
if ReleaseYear > 2015 → go left   (same split, after scaling: if z_score > 0.3)
```

The split condition is **equally valid** whether the value is `2015` or `0.3` — the relative ordering of data points doesn't change. Trees are **scale-invariant by design**.

Linear models are **not** scale-invariant — the magnitude of a feature directly determines the size of its learned coefficient, which makes features incomparable without scaling.

| Model Type | Needs StandardScaler? | Why |
|---|---|---|
| Linear Regression | ✅ Yes | Coefficients are directly proportional to feature magnitude |
| Ridge Regression | ✅ Yes | Same; regularization also penalizes large coefficients |
| Random Forest | ❌ No | Splits on thresholds — scale doesn't affect relative ordering |
| Gradient Boosting | ❌ No | Same tree-based logic — scale-invariant |

> Applying StandardScaler to tree models is **harmless but pointless** — the tree will find the exact same splits regardless. The code correctly skips it for RF and GB.

---

## 2. Feature Engineering

### 2.1 How We Measure the Relation Between Feature and Target

**Method: `SelectKBest` with F-Score**

```python
score_func = f_regression if task == "regression" else f_classif
selector = SelectKBest(score_func=score_func, k=K)
selector.fit(X, y)
```

**What is the F-Score here?**

It's an **ANOVA F-statistic** — a measure of how much variance in the *target* is explained by each *feature*, compared to the variance within groups.

| Task | Score Function | What It Measures |
|---|---|---|
| **Regression** | `f_regression` | Linear correlation between each feature and the continuous target (Pearson correlation → F-stat) |
| **Classification** | `f_classif` | ANOVA F-test — how much the feature's mean differs across target classes |

**Concrete example:**

- Feature: `SteamSpyOwners` — if games with more owners consistently have higher `RecommendationCount`, F-score will be high.
- Feature: `DemoCount` — if demo availability shows no systematic difference across popularity classes, F-score will be near zero.

**What we do NOT use and why:**

| Method | Why Not Used |
|---|---|
| **Mutual Information** | Better for non-linear relationships, but slower and less stable on large datasets |
| **Recursive Feature Elimination (RFE)** | Model-dependent, much more expensive — requires fitting a model at each step |
| **PCA** | Dimensionality reduction, not selection — loses feature interpretability |
| **Correlation Matrix** | ⚠️ Answers the **wrong question** — measures feature↔feature similarity, not feature↔target relevance; cannot rank features by predictive power |

> **⚠️ Why Correlation Matrix cannot do feature selection:**  
> A correlation matrix is a **Feature × Feature** table — the target column does not appear in it at all.  
> It tells you *"are two features redundant with each other?"* — not *"is this feature useful for predicting the target?"*  
> Example: `corr(SteamSpyOwners, SteamSpyPlayersEstimate) = 0.95` tells you those two are redundant — but says **nothing** about whether either one predicts `RecommendationCount`.  
> Correlation matrix is a valid tool for **removing duplicate features**, not for **ranking feature importance**.

> `SelectKBest` with `f_regression`/`f_classif` is the **standard academic baseline** — fast, interpretable, and task-appropriate. The F-scores are saved and plotted so you can visually inspect which features matter.

---

### 2.2 Why Select at Most 25 Features?

```python
K = min(k_max, X.shape[1])   # k_max = 25
```

**The `min()` is critical:** if the dataset has fewer than 25 features (after preprocessing), we take all of them. We never request more features than exist.

**Why 25 as the cap?**

| Consideration | Explanation |
|---|---|
| **Curse of Dimensionality** | More features = more dimensions = models need exponentially more data to generalize |
| **Overfitting risk** | With 50+ features, a model may memorize training noise rather than learn real patterns |
| **Computational cost** | Grid search over hyperparameters × many features = very slow training |
| **Interpretability** | 25 features is a human-readable number — you can reason about what drives predictions |
| **Signal vs. noise** | Beyond top-N features, F-scores drop sharply — low-scoring features add noise, not signal |

**The academic rationale:**  
Research consistently shows diminishing returns after ~15–30 features for tabular datasets of this scale. Selecting the top 25 by F-score captures the **majority of explainable variance** while protecting against overfitting.

> If you had 100+ features and removed the cap, models like Linear Regression would overfit badly. Ensemble methods (RF, GB) are more robust but still benefit from focused, relevant features.

---

## 3. Regression Models

### 3.1 Why These 4 Models?

The pipeline trains: **Linear Regression**, **Ridge Regression**, **Random Forest Regressor**, **Gradient Boosting Regressor**.

These represent a deliberate **spectrum from simple → complex**:

```
Simple ←—————————————————————————————————→ Complex
Linear Regression → Ridge → Random Forest → Gradient Boosting
```

| Model | Family | Key Strength | Key Weakness |
|---|---|---|---|
| **Linear Regression** | Linear | Maximally interpretable baseline | Assumes linear relationship; sensitive to collinearity |
| **Ridge Regression** | Regularized Linear | Handles collinearity; stable | Still linear — can't capture non-linear patterns |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearity, outlier-robust, parallel | Black-box, high memory |
| **Gradient Boosting** | Ensemble (Boosting) | Often best accuracy, learns residuals | Slow to train, prone to overfit if not tuned |

**Why not others?**

| Excluded Model | Reason |
|---|---|
| **SVR** | Extremely slow on large datasets; kernel adds another hyperparameter dimension |
| **KNN Regressor** | No trained model — slow at prediction; poor scaling to large N |
| **Neural Networks** | Overkill for tabular data; requires architecture search, more data, more tuning |
| **Lasso Regression** | Performs feature selection via L1 — redundant since we already selected with `SelectKBest` |
| **Decision Tree (single)** | High variance, prone to overfit — Random Forest is strictly better |

---

### 3.2 Regression Hyperparameter Grids & Rationale

#### Linear Regression — No Tuning

```python
model.fit(X_train_sc, y_train)
```

Linear Regression has **no hyperparameters** to tune. It has one objective (minimize OLS loss) and the math produces the unique optimal solution. Grid search would be pointless.

---

#### Ridge Regression

```python
"alpha": [0.1, 1.0, 10.0]
```

| Value | Meaning | Effect |
|---|---|---|
| `alpha = 0.1` | Very light regularization | Almost identical to plain Linear Regression |
| `alpha = 1.0` | Moderate (sklearn default) | Balanced trade-off — good starting point |
| `alpha = 10.0` | Strong regularization | Shrinks coefficients aggressively — more bias, less variance |

**Why these 3 values?** They span **2 orders of magnitude** — a logarithmic grid that efficiently covers the space from "barely regularized" to "heavily regularized". Testing 6 values would give more resolution but at 2× the grid search cost with diminishing benefit.

---

#### Random Forest Regressor

```python
"n_estimators": [100, 200, 300],
"max_depth":    [5, 10, 15],
"min_samples_leaf": [1, 2, 5]
```

This creates a **3 × 3 × 3 = 27-combination grid** evaluated with 3-fold CV → **81 model fits**.

| Hyperparameter | What It Controls |
|---|---|
| `n_estimators` | Number of trees in the forest |
| `max_depth` | How deep each tree can grow (controls complexity) |
| `min_samples_leaf` | Minimum samples required at each leaf node |

---

#### Gradient Boosting Regressor

```python
"n_estimators":  [100, 200, 300],
"learning_rate": [0.01, 0.05, 0.1],
"max_depth":     [3, 5, 7]
```

Again a **27-combination grid** with 3-fold CV → **81 model fits**.

---

### 3.3 How Hyperparameters Affect Regression Prediction

#### `alpha` (Ridge) — Bias-Variance Trade-off

```
Low alpha (0.1):   Low bias, High variance  → may overfit
High alpha (10.0): High bias, Low variance  → may underfit
Optimal alpha:     Sweet spot found by GridSearchCV
```

- Low `alpha`: Large coefficients allowed → model fits training data tightly
- High `alpha`: Coefficients shrunk toward zero → model generalizes better when features are collinear

---

#### `n_estimators` — Stability vs. Cost

```
100 trees: Faster, slightly noisy predictions
200 trees: Good balance
300 trees: More stable (law of large numbers), but 3× slower to train
```

More trees → predictions become more **stable** (averaging more independent learners). After ~200–300, gains plateau while cost keeps rising.

---

#### `max_depth` — Complexity Control

```
Shallow (depth=3–5):  High bias — underfits complex patterns
Deep   (depth=10–15): Low bias, high variance — may overfit noise
```

- **For Random Forest:** Shallower trees are fine because averaging many diverse trees handles bias.
- **For Gradient Boosting:** Conventionally shallower (`3–7`) because each tree only corrects the *residual* error. Deep trees in boosting overfit aggressively.

---

#### `min_samples_leaf` — Overfitting Safeguard

```
min_samples_leaf=1: Leaf can have a single sample → extreme overfitting
min_samples_leaf=5: Each prediction averages ≥5 training samples → smoother
```

Higher values produce smoother, less spiky predictions. If your model scores perfectly on train but poorly on test, increasing `min_samples_leaf` is the first thing to try.

---

#### `learning_rate` (Gradient Boosting) — Step Size

```
learning_rate=0.01: Very slow — needs more trees, but very stable
learning_rate=0.05: Moderate — usually the sweet spot
learning_rate=0.1:  Faster convergence — risks overshooting the optimum
```

**The key interaction:**

```
Lower learning_rate → Increase n_estimators (more iterations to converge)
Higher learning_rate → Fewer n_estimators needed
```

This is why both are tuned **together** in the same grid.

---

## 4. Classification Models

### 4.1 Why These 3 Models?

The pipeline trains: **Logistic Regression**, **Random Forest Classifier**, **Gradient Boosting Classifier**.

| Model | Family | Key Strength | Key Weakness |
|---|---|---|---|
| **Logistic Regression** | Linear | Fast, interpretable, probabilistic output | Assumes linear decision boundary |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearity, robust, parallelizable | Less accurate than boosting on many benchmarks |
| **Gradient Boosting** | Ensemble (Boosting) | Often state-of-the-art on tabular data | Slower training, more hyperparameters |

**Why not others?**

| Excluded Model | Reason |
|---|---|
| **SVM** | Kernel SVM is O(n²–n³) — prohibitively slow on large datasets |
| **KNN** | No generalization model — meaningless with many features (curse of dimensionality) |
| **Naive Bayes** | Assumes feature independence — game features (genre, platform, category) are highly correlated |
| **Decision Tree** | High variance, easily overfits — Random Forest is always better |
| **Neural Networks** | Tabular data at this scale doesn't benefit significantly; requires far more tuning |

---

### 4.2 Classification Hyperparameter Grids & Rationale

#### Logistic Regression

```python
"C": [0.1, 1.0, 10.0]
```

`C` is the **inverse of regularization strength** — the opposite convention from Ridge's `alpha`:

| `C` Value | Effect |
|---|---|
| `C = 0.1` | Strong regularization → simpler model, avoids overfitting |
| `C = 1.0` | Moderate (sklearn default) → balanced |
| `C = 10.0` | Weak regularization → model trusts training data more, risks overfit |

---

#### Random Forest & Gradient Boosting Classifiers

```python
# Random Forest
"n_estimators": [100, 200, 300],
"max_depth":    [5, 10, 15],
"min_samples_leaf": [1, 2, 5]

# Gradient Boosting
"n_estimators":  [100, 200, 300],
"learning_rate": [0.01, 0.05, 0.1],
"max_depth":     [3, 5, 7]
```

Identical grids to the regression variants — the hyperparameter **semantics are the same**. Only the scoring metric changes to `accuracy` for classification.

---

### 4.3 How Hyperparameters Affect Classification Prediction

#### `C` (Logistic Regression) — Decision Boundary Complexity

```
C = 0.1:  Wide margin, simple boundary → stable generalization
C = 10.0: Complex boundary, fits every training point → may misclassify test samples
```

If your game dataset has overlapping classes (e.g., "Niche" and "Popular" games with similar feature values), strong regularization (`C=0.1`) produces a robust classifier. Weak regularization makes the boundary too aggressive.

---

#### `n_estimators` — Voting Reliability

In classification, more trees means **majority voting** involves more independent opinions. The predicted class is more reliably the true majority.

**Effect on metrics:** Accuracy, Precision, Recall, and F1 all improve (up to a point) as `n_estimators` increases — then plateau.

---

#### `max_depth` — Decision Boundary Complexity

```
Shallow (depth=5):  Linear-ish boundaries — may miss complex class regions
Deep   (depth=15):  Very irregular boundaries — risk memorizing training labels
```

For **game popularity classification** (e.g., "Niche", "Popular", "Blockbuster"), the relationship is likely non-linear. A `max_depth` of 10 often hits the sweet spot.

---

#### `learning_rate` + `n_estimators` (Gradient Boosting)

Each boosting stage focuses on games the model **previously got wrong**.

```
Low LR (0.01) + Many trees (300):
  → Model slowly corrects mistakes → stable, resistant to overfit

High LR (0.1) + Few trees (100):
  → Model quickly adapts → may overshoot
```

**On class imbalance:** If `GamePopularity` is imbalanced (most games are "Niche"), boosting naturally focuses more iterations on misclassified minority classes — a subtle but real advantage.

---

#### `min_samples_leaf` — Class Purity in Leaves

```
min_samples_leaf=1: Perfect training accuracy → terrible generalization
min_samples_leaf=5: Smoother class probability estimates → reliable predictions
```

When `min_samples_leaf` is too small, the model becomes overconfident — assigns probability 0.99 to test samples but gets them wrong. Increasing this value produces calibrated, trustworthy predictions.

---

## Summary Table

| Decision | Choice Made | Alternative Considered | Reason for Choice |
|---|---|---|---|
| Text columns | Drop | NLP (TF-IDF / BERT) | Out of scope; classical ML can't consume raw strings |
| Categorical encoding | Pre-encoded as binary columns | OHE at runtime | Already done as indicator columns in the dataset |
| Missing numeric values | Median imputation | Mean / drop rows | Median is robust to skew; preserves all rows |
| Missing target values | Median (regression) / Mode (classification) | Drop rows | Preserves training examples with a reasonable estimate |
| Outlier handling | IQR clipping (Winsorization) | Remove rows / keep raw | Preserves row count; bounds extreme values without deletion |
| Skewed features | `log1p` transform | Box-Cox, sqrt | Handles zeros safely; industry standard for count data |
| Feature selection | `SelectKBest` + F-score | RFE, Mutual Info, PCA | Fast, interpretable, and task-appropriate |
| Feature count cap | k = 25 | All features | Prevents curse of dimensionality and overfitting |
| Regression models | LR, Ridge, RF, GB | SVR, KNN, NN, Lasso | Covers full simple→complex spectrum; excluded models are too slow or redundant |
| Classification models | LogReg, RF, GB | SVM, KNN, Naive Bayes | Fast, interpretable, state-of-the-art for tabular data |
| Hyperparameter search | GridSearchCV (3-fold CV) | RandomSearchCV, Bayesian opt | Exhaustive; 3-fold balances compute vs. reliability |
| Search metric (regression) | `neg_mean_absolute_error` | MSE, R² | MAE is more robust to outlier predictions |
| Search metric (classification) | `accuracy` | F1, AUC-ROC | Standard baseline metric; appropriate for initial evaluation |

---

*Report generated for: Online Games Popularity Prediction — Milestone 2*  
*Faculty of Computer & Information Sciences, Ain Shams University*
