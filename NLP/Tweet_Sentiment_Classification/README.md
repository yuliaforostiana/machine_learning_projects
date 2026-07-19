# Tweet Sentiment Classification: Bag-of-Words vs. TF-IDF

## 📌 Overview

This project builds a text classifier that predicts tweet sentiment (positive / negative / neutral), directly comparing two classic text vectorization approaches — Bag-of-Words and TF-IDF — across three model families (Logistic Regression, Decision Tree, XGBoost). Beyond picking a "winning" model, the project digs into *why* certain vectorization/model combinations overfit, which words actually drive each model's predictions, and where the final classifier systematically fails — closing with concrete, evidence-based recommendations for improving it further.

## 🎯 Problem Statement

Classifying sentiment from short, informal social media text is challenging: tweets are short, full of slang, repeated characters, emphasis marks, and ambiguous or mixed emotional tone. A good sentiment classifier needs text preprocessing that preserves meaningful signal (e.g., negation words like "not") while reducing noise, a vectorization method that captures the vocabulary efficiently, and a model that generalizes to genuinely ambiguous cases rather than just memorizing training examples.

## 🎯 Goal

- Clean and prepare a large, informal tweet dataset for text classification, including noise reduction (URLs, repeated characters, punctuation runs) and linguistically informed stopword handling.
- Vectorize text using both Bag-of-Words and TF-IDF, with matched preprocessing, to enable a fair comparison.
- Train and compare Logistic Regression, Decision Tree, and XGBoost classifiers (with hyperparameter tuning) on each vectorization method.
- Identify which words most influence the chosen model's predictions, and assess whether they make intuitive sense.
- Analyze misclassified examples to understand the model's failure modes and propose concrete next steps for improvement.

## 🔍 Approach

### 1. Data Loading & Cleaning
- Loaded the Kaggle Tweet Sentiment Extraction dataset and dropped records with missing `text`/`selected_text` values.
- Profiled class balance (`sentiment`: positive/negative/neutral) and text length distribution, both overall and per sentiment class.

### 2. Exploratory Data Analysis
- Found sentiment classes to be close to balanced, with a slight over-representation of the neutral class.
- Found that neutral and negative posts tend to run longer (up to 141 characters) than positive posts, and that roughly half of positive/negative posts exceed the median length of neutral posts — a subtle length-based signal potentially useful for classification.

### 3. Text Preprocessing & Bag-of-Words Vectorization
- Built a custom `preprocess_text()` pipeline: lowercasing, URL removal, normalizing repeated punctuation (`!!!`→`!`, `***`→`star`), collapsing repeated characters (`sooo`→`soo`), stopword removal (explicitly preserving negation words `not`/`no`/`nor`, since they carry strong sentiment signal), tokenization, and stemming (`SnowballStemmer`).
- Analyzed word frequency and vocabulary coverage to justify vectorizer settings: `max_features=5000` was chosen because it covers roughly 85–90% of the corpus while limiting noise from rare tokens.
- Vectorized with `CountVectorizer` (`max_features=5000`, `ngram_range=(1,2)` to capture negation phrases like "not bad").

### 4. Model Training & Comparison (Bag-of-Words)
- Trained and compared three classifiers on the same BoW-vectorized train/test split (stratified 70/30):
  - **Logistic Regression** (`sag` solver)
  - **Decision Tree** (tuned via `RandomizedSearchCV`, 50 iterations, `f1_weighted` scoring)
  - **XGBoost** (tuned via `RandomizedSearchCV`, 20 iterations, `f1_weighted` scoring, multi-class objective)
- Evaluated all three with a shared `classify_analysis` function (confusion matrix, F1-score, AUROC) on both training and validation data.

### 5. Feature Importance Analysis
- Extracted and inspected the top 20 most influential tokens for the selected model, checking whether they aligned with intuitive sentiment-bearing words.

### 6. TF-IDF Comparison
- Repeated the exact same preprocessing and train/test split with `TfidfVectorizer` (matched settings) to enable an apples-to-apples comparison against Bag-of-Words.
- Retrained the best-performing model (XGBoost) on TF-IDF vectors and compared performance and top influential tokens against the BoW version.

### 7. Error Analysis
- Isolated misclassified validation examples for the TF-IDF model and broke down error rates by true sentiment class and by specific confusion pairs (e.g., negative predicted as neutral).

## 📊 Results

| Model | Vectorization | Notes |
|---|---|---|
| Logistic Regression | Bag-of-Words | Strong training performance but overfit — noticeably weaker on validation |
| Decision Tree (tuned) | Bag-of-Words | Lowest performance of the three models |
| **XGBoost (tuned)** | **Bag-of-Words** | Some train/validation gap, but the most stable F1-score and AUROC — **selected as final model** |
| XGBoost (tuned) | TF-IDF | Higher training scores but clearer overfitting on validation; also notably slower to train |

**Key findings:**
- **XGBoost on Bag-of-Words vectors** was selected as the final model — while Logistic Regression and XGBoost had comparable training performance, XGBoost generalized more consistently to validation data.
- **TF-IDF did not outperform Bag-of-Words** for this task: it showed clear overfitting and took substantially longer to train, making Bag-of-Words the more practical choice here despite TF-IDF's typically stronger reputation for text classification.
- **Top influential words were consistent and intuitive** across both vectorization methods — the top 4 tokens (in varying order) were shared between BoW and TF-IDF feature importances, reinforcing that the model is picking up genuine sentiment signal rather than noise.
- **Error analysis revealed a clear failure pattern:** the model most often confuses negative posts with neutral (~38.6% error rate for the negative class) and, to a lesser extent, positive with neutral (~28.5%). Overall validation error rate is about 30%. Misclassified examples tend to be short, ambiguous, or only weakly emotionally expressive — suggesting the model struggles most with subtle sentiment rather than clearly-worded posts.

**Proposed improvements** (from the error analysis): 
1. Better handling of slang, abbreviations, and spelling variants common in tweets.
2. Revisit Logistic Regression with stronger regularization, since it performed comparably to XGBoost on Bag-of-Words but overfit — regularization could close that gap.
3. Further tune `TfidfVectorizer` parameters (vocabulary size, n-gram range) now that a preprocessing baseline exists, rather than only matching the BoW configuration.

## 🛠️ Tech Stack

- **Language:** Python
- **NLP & text processing:** `nltk` (tokenization, stopwords, `SnowballStemmer`), `re` (regex-based text cleaning)
- **Vectorization:** `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`)
- **Modeling:** `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `RandomizedSearchCV`, `LabelEncoder`), `XGBoost`
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`
- **Evaluation:** confusion matrix, F1-score, AUROC, `classification_report`

## 📁 Repository Structure

├── tweet_sentiment_classification.ipynb   # Full preprocessing, vectorization, modeling, and error analysis notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk xgboost
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
jupyter notebook tweet_sentiment_classification.ipynb
```

> Note: the dataset (`tweet_sentiment_train.csv.zip`) is from the Kaggle [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data) competition and is not included in this repository.

## 📈 Next Steps

- Address the identified failure modes: expand preprocessing to normalize common slang/abbreviations and correct spelling variants before stemming.
- Re-test a regularized Logistic Regression on Bag-of-Words vectors as a lighter-weight alternative to XGBoost, given its comparable raw performance.
- Explore word embeddings (Word2Vec, GloVe) or transformer-based encoders (e.g., DistilBERT) as a further upgrade path beyond sparse Bag-of-Words/TF-IDF representations, particularly to better handle the short, ambiguous posts identified in the error analysis.
