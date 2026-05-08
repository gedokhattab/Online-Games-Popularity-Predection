# 🎮 Online Games Popularity Predictor

A terminal‑first ML pipeline that predicts:
- **RecommendationCount** → regression
- **GamePopularity** → classification

## Repository layout
```
Project/
├─ Datasets/            # CSV files
│   ├─ idlist.csv
│   ├─ regression_train_data.csv
│   └─ classification_train_data.csv
├─ Models/              # generated artifacts
├─ Plots/
├─ Tables/
├─ Src/                 # source code
│   ├─ app.py      # menu UI
│   ├─ train.py    # pipeline driver
│   ├─ test.py     # inference script
│   ├─ preprocessing.py
│   ├─ feature_engineering.py
│   ├─ regression_model_training.py
│   ├─ classification_model_training.py
│   └─ colors.py
├─ launch.bat      # one-click start
└─ README.md
```

## Prerequisites
- Python ≥ 3.10
- Windows CMD (or any terminal supporting ANSI colours)
- Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Quick start
```bash
# from the repository root
launch.bat
```
Select a number from the menu:
1 – Train regression
2 – Train classification
3 – Train both
4 – Test regression
5 – Test classification
6 – Exit

All artifacts are saved under `Models/`, visualizations under `Plots/`, and tables under `Tables/`.

## Customisation
Edit colour definitions in `Src/colors.py` if you want a different theme.

## License
This project is **unlicensed** – you may not modify, or distribute it without permission.

## Authors
- Khattab Mohamed
