# Book Recommendation Systems on Goodbooks-10k

## 📌 Overview

This project implements and compares four recommender system architectures — Vector Space Model, Two-Tower retrieval, Concat-based ranking (Neural Collaborative Filtering), and a two-stage Retrieval→Ranking pipeline — on the real-world **Goodbooks-10k** dataset (millions of book ratings from tens of thousands of users). Rather than treating recommendation as a single-model problem, the project builds each architecture from scratch in PyTorch, benchmarks them with ranking-specific metrics, and closes with a critical, evidence-based analysis of *why* offline recall is lower than one might hope on real, sparse data — and what would genuinely improve it.

## 🎯 Problem Statement

Goodbooks-10k has no built-in genre labels, highly sparse user-item interactions, and only a handful of usable content features once genres are engineered from noisy user tags — a realistic setting quite unlike toy datasets used to teach recommender concepts. The goal is to see how standard recommender architectures actually perform under these real-world constraints: how retrieval scales across thousands of candidate items, why content-based similarity alone is often insufficient, and why standard offline ranking metrics like Recall@K tend to look modest even for reasonable models.

## 🎯 Goal

- Engineer usable genre features for books from noisy, user-generated tags, since no ground-truth genre labels exist in the raw data.
- Implement a content-based Vector Space Model (cosine similarity in genre space) as a simple, interpretable baseline.
- Train a Two-Tower neural retrieval model (learned user/item embeddings with negative sampling) as a scalable alternative.
- Train a Concat-based (early-fusion) NCF ranking model that can capture richer user-item interactions at the cost of not being indexable.
- Combine Two-Tower retrieval with NCF ranking into a realistic two-stage recommendation pipeline, mirroring how large-scale production systems operate.
- Evaluate all approaches with **Recall@K** (a ranking-appropriate metric) rather than RMSE, and critically diagnose the sources of its offline limitations.

## 🔍 Approach

### 1. Feature Engineering: Genres from Tags
- Since Goodbooks-10k has no genre column, mapped a curated list of 12 canonical genres (fantasy, romance, mystery, thriller, etc.) onto Goodreads user tags, producing a binary **book × genre** matrix used as the only content features for all models.

### 2. Data Subsampling for Tractable Training
- Reduced ~6M ratings to a dense, trainable subset: top 1,500 most-rated books, users with ≥20 ratings, and a random sample of 2,000 of those active users — preserving interaction density while keeping CPU-friendly training times.
- Defined "likes" as ratings ≥ 4, and built train/validation splits at the interaction level, tracking each user's already-seen books to avoid leaking them into evaluation candidates.

### 3. Ranking Metric: Recall@K
- Implemented `recall_at_k()` from scratch: for each validation user, score all candidate books, exclude already-seen books, take the top-K, and measure overlap with the user's actual validation-set likes — a more realistic quality signal for recommendation than RMSE, which only measures rating-prediction accuracy rather than ranking quality.

### 4. Model 1 — Vector Space Model (content-based baseline)
- Represented each book as an L2-normalized genre vector; represented each user as a rating-weighted average of the genre vectors of books they liked.
- Recommended via cosine similarity (equivalent to a dot product after normalization) between user and item vectors.

### 5. Model 2 — Two-Tower Retrieval
- Built a `TwoTower` model with a learned user embedding tower and an item tower (genre features → MLP), both L2-normalized, trained with in-batch negative sampling and `BCEWithLogitsLoss`, using a temperature-scaled dot product to sharpen the sigmoid's confidence range.
- Item vectors are computed once and reused for fast retrieval — a late-fusion design suited to large-scale, indexable retrieval (e.g., via FAISS in production).

### 6. Model 3 — Concat-based Ranking (NCF)
- Built an `NCF` model using early fusion: concatenates the user embedding with the item's genre features and passes the combined vector through an MLP to produce a single relevance logit per (user, item) pair.
- More expressive than Two-Tower (can model direct user-item interactions) but not indexable — every candidate pair must be scored explicitly, making it suitable only for ranking a short candidate list, not full-catalog retrieval.

### 7. Two-Stage Pipeline: Retrieval → Ranking
- Combined both models as production systems typically do: Two-Tower quickly retrieves a shortlist of candidates (e.g., top 50) from the full catalog, then NCF re-ranks that shortlist for the final top-K recommendations — balancing retrieval speed with ranking precision.

## 📊 Results

| Model | Approach | Recall@10 | Notes |
|---|---|---:|---|
| Vector Space Model | Content-based, cosine similarity | Low | Recommends "the right genre direction" but often the wrong specific book among similar-genre titles |
| Two-Tower | Learned embeddings, negative sampling | **Best of the individually-evaluated models** | Closest alignment with the example user's actual preferences |
| NCF | Early-fusion MLP ranking | Comparable, requires much more compute per query | Not indexable — every (user, item) pair scored explicitly |
| Two-Stage Pipeline | Two-Tower retrieval → NCF ranking | Combines strengths of both | For the example user, final NCF re-ranking skewed toward romance/sci-fi, somewhat diverging from the user's stronger young-adult/fantasy profile |

**Key findings:**
- **All models showed modest Recall@10** on this real dataset — traced to several compounding factors: only 12 genre features to describe books (causing many books to collapse to near-identical vectors), high interaction sparsity, and validation likes representing only a partial view of what a user *might* actually enjoy (so a "correct" recommendation the user hasn't rated yet still counts as a miss).
- **Two-Tower generalized best** among the individually evaluated models for the example user examined in detail, though the project explicitly notes this should be validated across multiple representative users before drawing firm conclusions.
- **The two-stage pipeline's ranking stage can shift recommendations away from a user's dominant genres** if the ranking model isn't well-calibrated — a reminder that combining retrieval and ranking doesn't automatically guarantee the final list matches user intent better than either stage alone.
- **Vector Space Model recommendations lacked diversity**, tending to surface books with nearly identical genre vectors rather than a varied slate.

### Reflection questions (from the project's theoretical section)
- **Why is Recall@10 low?** Limited content features (12 genres), high sparsity, and incomplete validation coverage of true user preferences all contribute.
- **How to improve without changing architecture?** Add richer book metadata (author, publication year, page count), aggregate signals (average rating, rating count, % 5-star), semantic embeddings (BERT/Sentence-Transformers on book descriptions), and user-level behavioral features.
- **Diversity:** showing 10 near-identical fantasy books back-to-back risks user fatigue — diversity can be enforced via re-ranking penalties, genre-quota sampling, or maximal marginal relevance (MMR)-style post-processing.
- **Cold start:** a brand-new book with zero ratings can still be recommended by the Vector Space Model and Two-Tower's item tower (both rely only on content features), but NCF's early-fusion approach and any purely collaborative signal would struggle without interaction history.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** `pandas`, `numpy`
- **Modeling:** PyTorch (`torch.nn.Embedding`, custom `TwoTower` and `NCF` modules, `BCEWithLogitsLoss`, `Adam` optimizer)
- **Evaluation:** custom `recall_at_k()` ranking metric implementation

## 📁 Repository Structure

├── book_recommendation.ipynb   # Full feature engineering, model implementations, and evaluation notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy torch
jupyter notebook book_recommendation.ipynb
```

Data can be loaded three ways, in order of preference:
1. **Kaggle API** — download `kaggle.json` from your Kaggle account and use it to fetch the [Goodbooks-10k dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) directly.
2. **Manual download** — download the dataset from Kaggle and place `ratings.csv`, `books.csv`, `book_tags.csv`, and `tags.csv` alongside the notebook.
3. **GitHub mirror (automatic fallback)** — if the files aren't found locally, the notebook fetches them automatically from the original author's GitHub mirror.

> A GPU is optional but speeds up Two-Tower and NCF training.

## 📈 Next Steps

- Enrich item features beyond the 12 engineered genres — author, publication year, aggregate rating statistics, and semantic text embeddings (e.g., BERT on book descriptions) to give the models more to work with than a sparse genre vector.
- Evaluate across a representative sample of multiple users (not just one illustrative example) before drawing conclusions about which architecture "wins."
- Add explicit diversity control (e.g., MMR re-ranking or genre-quota sampling) to the final recommendation list, particularly for the two-stage pipeline.
- Test the cold-start behavior explicitly by adding synthetic zero-interaction books and confirming which models can still surface them.
