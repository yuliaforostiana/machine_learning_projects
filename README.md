# Machine Learning & Data Science Projects

A collection of end-to-end machine learning and data analysis projects, spanning exploratory data analysis, classical ML (regression, classification, clustering), time series forecasting, deep learning, NLP, recommender systems, and LLM agent workflows. Each project lives in its own folder with a dedicated README covering the problem, approach, results, and tech stack.

## 📁 Projects

### Exploratory Data Analysis
| Project | Description | Stack |
|---|---|---|
| [`credit_eda/`](https://github.com/yuliaforostiana/machine_learning_projects/blob/main/EDA/credit_eda.ipynb) | EDA on a 300K+ record loan application dataset to identify factors driving credit default risk. | pandas, seaborn |

### Regression
| Project | Description | Stack |
|---|---|---|
| [`medical_charges_prediction/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Supervised_Learning/Linear_Regression/medical_charges_prediction) | Simple linear regression (OLS from scratch, gradient descent, scikit-learn) predicting medical charges from age. | numpy, scikit-learn |
| [`car_price_prediction/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Supervised_Learning/Linear_Regression/car_price_prediction) | Multivariate linear regression with statistical feature selection and diagnostics (VIF, heteroscedasticity tests) for used car pricing. | statsmodels, scikit-learn |

### Time Series Forecasting
| Project | Description | Stack |
|---|---|---|
| [`store_time_series_analysis/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Supervised_Learning/Time_Series_Analysis) | Benchmark of 8+ forecasting models (Prophet, ARIMA, XGBoost, RNN, naive baselines) for daily retail sales. | darts, statsmodels |
| [`lstm_airline_passengers_forecast/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Deep_Learning/Airline_passenger_forecast) | Custom LSTM built in PyTorch for the classic airline passengers forecasting problem, with a seasonality diagnostic experiment. | PyTorch |

### Classification
| Project | Description | Stack |
|---|---|---|
| [`bank_churn_prediction/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Supervised_Learning/Bank_Customer_Churn_Prediction) | Progressive model benchmarking (logistic regression → decision trees → kNN → XGBoost/LightGBM) for churn prediction, with a Kaggle submission pipeline. | scikit-learn, XGBoost, LightGBM, hyperopt |
| [`customer_segmentation_classification/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Supervised_Learning/Customer_Segmentation) | Imbalanced multiclass classification (One-vs-Rest / One-vs-One) with SMOTE-family resampling comparison. | scikit-learn, imbalanced-learn |
| [`tweet_sentiment_classification/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/NLP/Tweet_Sentiment_Classification) | Sentiment classification comparing Bag-of-Words vs. TF-IDF vectorization across three model families, with error analysis. | nltk, scikit-learn, XGBoost |

### Clustering
| Project | Description | Stack |
|---|---|---|
| [`customer_segmentation_clustering/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Unsupervised_learning/%20Clustering_Marketing_Segmentation) | Customer personality segmentation benchmarking K-Means, hierarchical clustering, and Gaussian Mixture Models, with feature engineering. | scikit-learn, scipy |

### Deep Learning
| Project | Description | Stack |
|---|---|---|
| [`cnn_mnist_classification/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/CV/MNIST_Classification) | CNN (TinyVGG) built from scratch in PyTorch for digit classification, with CPU/GPU benchmarking and conv-layer hyperparameter experiments. | PyTorch, torchvision |

### Recommender Systems
| Project | Description | Stack |
|---|---|---|
| [`book_recommendation/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/Recommendation_Systems) | Four recommender architectures (Vector Space Model, Two-Tower, NCF, two-stage Retrieval→Ranking) benchmarked on real book rating data. | PyTorch |

### LLMs & Agents
| Project | Description | Stack |
|---|---|---|
| [`langchain_prompts_and_agents/`](https://github.com/yuliaforostiana/machine_learning_projects/tree/main/LLM/%20Prompting_Agentic_Workflow) | Parameterized prompting and ReAct agent workflows (web search, Python execution) for research and business forecasting tasks. | LangChain, Hugging Face |

## 🛠️ General Notes

- Each project folder contains its own `README.md` with a full problem statement, methodology, results, and setup instructions.
- Datasets are not committed to this repository; each project README links to or names the original data source.
- Most projects are implemented as Jupyter notebooks; see each folder's README for the exact file(s) and how to run them.

## 📬 About

This repository collects independent projects exploring different areas of the machine learning workflow — from data cleaning and classical statistics to deep learning and modern LLM-based agents — each treated as a self-contained case study.
