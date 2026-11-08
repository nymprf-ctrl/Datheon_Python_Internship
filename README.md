# Machine Learning Projects: Classification, Regression, Ensembles and NLP

Four end-to-end machine learning projects, each delivered as a Jupyter notebook with executed outputs, plots and written findings. Together they cover supervised classification, regression with regularization, ensemble learning on imbalanced data, and text classification.

## Projects at a glance

| # | Project | Notebook | Dataset | Models | Headline result |
|---|---------|----------|---------|--------|-----------------|
| 1 | Iris species classification | `iris_classification.ipynb` | `iris.csv` (150 rows) | Logistic Regression, Decision Tree | Test accuracy 93.3% / 96.7% |
| 2 | California house price prediction | `housing_regression.ipynb` | `housing.csv` (20,640 rows) | OLS, Ridge, Lasso | Test R² 0.661, RMSE about 66,660 |
| 3 | Telecom customer churn prediction | `churn_prediction.ipynb` | `Telco-Customer-Churn.csv` (7,043 rows) | Random Forest, XGBoost | Test ROC-AUC 0.836 to 0.847 |
| 4 | Review sentiment classification | `sentiment_classifier.ipynb` | `movie_reviews.csv` (2,000 rows) | Multinomial Naive Bayes, Logistic Regression | Test accuracy 80.8% / 84.0% |

## 1. Iris classification

**Goal:** classify iris species from four flower measurements.

- Exploratory pairplot and correlation heatmap
- 80/20 stratified split with `StandardScaler` fitted on the training set only
- Logistic Regression and a depth-limited Decision Tree
- Accuracy, precision, recall and F1, confusion matrix heatmaps, and decision boundary plots on petal length and petal width
- 5-fold cross-validation as a more reliable comparison than a single 30-sample test set

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.967 | 0.970 | 0.967 | 0.967 |

Setosa is separated perfectly by both models. Nearly all errors involve versicolor and virginica.

## 2. California housing regression

**Goal:** predict `median_house_value` and compare OLS, Ridge (L2) and Lasso (L1).

- Median imputation for missing `total_bedrooms`, learned on the training set only
- Feature engineering: per-household ratios (clipped at training percentiles) and mean-centered interaction terms
- Multicollinearity handling with correlation heatmaps and VIF
- `StandardScaler` and one-hot encoding of `ocean_proximity` inside a single pipeline
- Ridge and Lasso tuned with 5-fold `GridSearchCV`
- Predicted vs actual plots, residual distribution curves, residuals vs predicted, and Q-Q plots

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS (raw features) | 70,059 | 50,670 | 0.625 |
| OLS (engineered) | 66,659 | 48,100 | 0.661 |
| Ridge | 66,671 | 48,111 | 0.661 |
| Lasso | 66,673 | 48,113 | 0.661 |

Feature engineering gives the real improvement. Ridge and Lasso perform almost the same as OLS because there are many rows and few features. The target is capped at 500,001, which limits what a linear model can achieve.

## 3. Customer churn prediction

**Goal:** predict which telecom customers will churn, using ensemble models on imbalanced data (about 26.5% churn).

- `TotalCharges` converted to numeric; the 11 blank rows all have `tenure = 0` and are filled with 0
- `OneHotEncoder` for categorical features
- Imbalance handled two ways: class weighting (`class_weight`, `scale_pos_weight`) and SMOTE inside the cross-validation folds
- Random Forest and XGBoost tuned with 5-fold stratified `RandomizedSearchCV`
- ROC and precision-recall curves, confusion matrices, decision threshold tuning, top-10 feature importances, and permutation importance

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Random Forest + class weights | 0.8479 | 0.8431 |
| Random Forest + SMOTE | 0.8430 | 0.8363 |
| XGBoost + scale_pos_weight | 0.8495 | 0.8470 |
| XGBoost + SMOTE | 0.8478 | 0.8462 |

All four models exceed the 0.80 ROC-AUC target. The strongest churn drivers are contract type and tenure.

> **Runtime:** the hyperparameter search takes roughly 10 to 15 minutes on a single CPU.

## 4. Sentiment classification

**Goal:** classify review text as positive or negative using TF-IDF features.

- Text cleaning: lowercasing, HTML and URL removal, contraction expansion, punctuation removal, stopword removal, lemmatization
- `TfidfVectorizer(max_features=5000)`
- Multinomial Naive Bayes and Logistic Regression, tuned with cross-validation
- Comparison of preprocessing choices (raw text, stopword lists, negation handling, bigrams)
- `predict_sentiment()`, `compare_models()` and `interactive_tester()` for testing custom phrases

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Multinomial Naive Bayes | 0.808 | 0.802 | 0.892 |
| Logistic Regression | 0.840 | 0.847 | 0.925 |

**Data note:** the supplied `sentiment_reviews.csv` contains only 10 reviews, which is too few to train a classifier. The models are trained on the 2,000-review NLTK Movie Reviews corpus (`movie_reviews.csv`), and the 10 product reviews are used as a small out-of-domain check. Movie-trained models transfer poorly to product reviews (about 5 to 6 of 9 correct), so a product-review model should be trained on a labeled product-review dataset.

## Repository structure

Keep each notebook in the same folder as its data file. The notebooks load data using relative file names.

```
.
├── README.md
├── requirements.txt
├── iris_classification.ipynb
├── iris.csv
├── housing_regression.ipynb
├── housing.csv
├── churn_prediction.ipynb
├── Telco-Customer-Churn.csv
├── sentiment_classifier.ipynb
├── movie_reviews.csv
└── sentiment_reviews.csv
```

## Getting started

1. Clone the repository and open a terminal in its folder.

   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. Create a virtual environment and install the dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start Jupyter from this folder and open a notebook.

   ```bash
   jupyter notebook
   ```

4. Run the cells from top to bottom (Kernel → Restart & Run All). The notebooks already contain executed outputs, so they can also be read directly on GitHub without running anything.

The sentiment notebook downloads the NLTK `stopwords` and `wordnet` resources on first run, so it needs an internet connection the first time.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, NLTK, SciPy, seaborn, matplotlib, Jupyter.

## Tested versions

The notebooks were executed with Python 3.12, pandas 3.0, NumPy 2.5, scikit-learn 1.9, imbalanced-learn 0.14, XGBoost 3.4, seaborn 0.13, matplotlib 3.11 and NLTK 3.10. `requirements.txt` sets minimum versions. If a cell fails after an install, upgrade the packages with `pip install -U -r requirements.txt` first.

## Reproducibility

All random operations use `random_state=42`, so a rerun reproduces the reported numbers on the same library versions.
