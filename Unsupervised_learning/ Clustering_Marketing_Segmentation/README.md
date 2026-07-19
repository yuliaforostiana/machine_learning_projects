# Customer Personality Analysis: Clustering for Marketing Segmentation

## 📌 Overview

This project applies unsupervised clustering to a marketing campaign dataset to uncover meaningful customer segments — grouping customers by income, spending behavior, and purchasing channel to support targeted marketing decisions. Rather than settling on a single algorithm, the project systematically benchmarks K-Means, hierarchical clustering (single-linkage and Ward), and Gaussian Mixture Models across raw, scaled, outlier-treated, and feature-engineered versions of the data to find the segmentation that is both statistically sound and practically interpretable.

## 🎯 Problem Statement

Businesses that understand their customer base can focus marketing spend on the segments most likely to respond, rather than targeting their entire customer list uniformly. The challenge with real customer data is that it mixes numeric, categorical, and date fields, contains missing values and outliers, and the "right" number of clusters and the "right" preprocessing choices (scaling, outlier handling, feature engineering) all materially change what segments emerge — so the segmentation itself has to be validated, not just computed once and trusted.

## 🎯 Goal

- Clean and prepare a mixed-type marketing dataset for clustering, handling missing values and encoding categorical and date features appropriately.
- Benchmark multiple clustering algorithms (K-Means, hierarchical/agglomerative clustering, Gaussian Mixture Models) and preprocessing choices (raw vs. scaled, with vs. without outlier treatment) using the silhouette score.
- Determine the optimal number of clusters via the Elbow method and validate it against silhouette scores.
- Engineer higher-level behavioral features (total spend, spending by category, purchase-channel ratios, campaign responsiveness) to test whether they produce clearer, more actionable segments.
- Select and interpret the final segmentation in terms a marketing team could act on.

## 🔍 Approach

### 1. Data Preparation
- Imputed missing `Income` values using the median within `Education` × `Marital_Status` groups, rather than a global median, to better reflect each customer's likely income given their social profile.
- One-hot encoded `Education` and `Marital_Status` (no natural order, so ordinal/label encoding was avoided to prevent introducing artificial distance relationships).
- Decomposed the enrollment date (`Dt_Customer`) into a year, day-of-week, weekend flag, and cyclical (sine/cosine) month encoding to avoid treating month as a false linear scale.

### 2. Baseline Clustering & Scaling Comparison
- Ran an initial K-Means clustering (k=3) on the cleaned, encoded data and evaluated it with the silhouette score.
- Visualized clusters across multiple feature combinations (income, recency, purchase channels; income, spending, family composition) and profiled each cluster's characteristics.
- Repeated the same clustering on standardized (`StandardScaler`) data and compared silhouette scores and cluster composition — finding that unscaled clustering (implicitly weighted toward `Income`, given its larger numeric range) produced a notably higher silhouette score, though this reflected income dominating the distance metric rather than necessarily better-separated segments.

### 3. Outlier Treatment
- Identified genuine outliers (as opposed to long right tails reflecting a genuinely wealthier customer segment) in three key fields: `Year_Birth`, `Income`, and `MntMeatProducts`.
- Applied targeted treatment: filtered unrealistic birth years, winsorized extreme income values at the 1st/99th percentiles, and log-transformed `MntMeatProducts`.

### 4. Optimal Cluster Count & Algorithm Comparison
- Used the **Elbow method** (sum of squared distances vs. k) to identify 3–4 as the optimal cluster count on the outlier-treated data, for both unscaled and scaled versions.
- Compared K-Means against **hierarchical/agglomerative clustering** (single-linkage and Ward methods, visualized via truncated dendrograms and cut into flat clusters with `fcluster`) and **Gaussian Mixture Models**, evaluating each via silhouette score and cluster interpretability.

### 5. Feature Engineering & Final Comparison
- Engineered new behavioral features: `TotalSpent`, category-level spend (`SpentOnFood`, `SpentOnLuxury`), `TotalPurchases`, channel-usage ratios (`OnlineRatio`, `DealRatio`), aggregate `CampaignResponse`, household composition (`TotalKids`, `HasPartner`, `Alone`), and `Age`.
- Re-ran K-Means, Ward hierarchical clustering, and Gaussian Mixture Models on both unscaled and scaled versions of this enriched feature set, comparing silhouette scores and cluster profiles across all combinations.

## 📊 Results

| Stage | Configuration | Silhouette Score | Notes |
|---|---|---:|---|
| Baseline | K-Means (k=3), unscaled, raw features | Higher (income-dominated) | Clusters split primarily by income |
| Baseline | K-Means (k=3), standardized | ~3.5x lower than unscaled | More balanced feature contribution, but less separable |
| Outlier-treated | K-Means (k=3), unscaled | **0.55** | Most even, interpretable 3-cluster split |
| Outlier-treated | K-Means (k=4), unscaled | Slightly below 0.55 | Splits the mid-income segment into two behaviorally distinct groups |
| Outlier-treated | Hierarchical (single-linkage) | Comparable to K-Means | Clusters formed as income/spending "bands" rather than cohesive segments |
| Feature-engineered | K-Means / Ward (scaled) | Improved over baseline enriched-feature clustering | Best-performing configurations overall |

**Key findings:**
- **Income is the dominant clustering signal** throughout — unscaled data consistently produces higher silhouette scores because income's large numeric range implicitly weights the distance metric, but this can mask more nuanced, multi-feature segment structure.
- Outlier treatment on `Year_Birth`, `Income`, and `MntMeatProducts` substantially clarified cluster visualizations and produced more even, interpretable segments without materially hurting the silhouette score.
- Both k=3 and k=4 produced sensible segmentations after outlier treatment: k=3 gives a clean high/mid/low spend split, while k=4 further splits the mid-tier segment into "parents with children" (higher category diversity) vs. "parents with teens" (more discount-driven).
- **Feature engineering was the most impactful lever tested** — aggregating raw purchase/spend columns into behavioral ratios and totals (`TotalSpent`, `OnlineRatio`, `DealRatio`, `CampaignResponse`, etc.) improved silhouette scores on both scaled and unscaled data and shifted total spend (not just income) into a key differentiator between clusters.
- **Final recommendation:** the **K-Means clustering on unscaled engineered features**, or the **Ward hierarchical clustering on scaled engineered features**, produced the most logical and actionable customer segments among all combinations tested.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scipy.stats`
- **Preprocessing:** `scikit-learn` (`OneHotEncoder`, `StandardScaler`, `RobustScaler`)
- **Clustering:** `scikit-learn` (`KMeans`, `DBSCAN`, `GaussianMixture`), `scipy.cluster.hierarchy` (`linkage`, `dendrogram`, `fcluster`)
- **Evaluation:** silhouette score (`sklearn.metrics`), Elbow method

## 📁 Repository Structure

├── customer_segmentation_clustering.ipynb   # Full clustering analysis and comparison notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy
jupyter notebook customer_segmentation_clustering.ipynb
```

> Note: the dataset (`marketing_campaign.csv`, tab-separated) is not included in this repository.

## 📈 Next Steps

- Validate cluster stability across multiple random seeds and bootstrap samples, since K-Means results can vary with initialization.
- Test DBSCAN (imported but not benchmarked in this notebook) as a density-based alternative, particularly useful if segments have non-convex shapes or meaningful noise/outlier points.
- Translate the final segments into concrete marketing personas with actionable recommendations (e.g., channel preference, discount sensitivity) for each cluster, validated against real campaign response data (`Response`, `AcceptedCmp1-5`).
