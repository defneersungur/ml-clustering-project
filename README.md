# 🧠 ML Clustering Project

A machine learning project consisting of two parts: a supervised classification task on healthcare data, and an unsupervised customer segmentation task using clustering algorithms.

## 📁 Project Structure

| File | Description |
|---|---|
| `preprocessing_pipeline.py` | Data cleaning, encoding, scaling pipeline for Part 1 |
| `models.py` | KNN, ANN, and Random Forest classification models for Part 1 |
| `clustering_models.py` | K-Means, DBSCAN, and Mean Shift clustering for Part 2 |

## 📌 Part 1 — Classification: Normal Weight Prediction

**Goal:** Predict whether a patient is normal weight or not based on height and weight data.

**Approach:**
- Exploratory data analysis (EDA) with visualizations
- Height unit conversion (EU metric → inches)
- Outlier removal using IQR method
- Feature scaling with StandardScaler pipeline
- Three different models trained and compared:
  - K-Nearest Neighbors (KNN) with distance weighting
  - Artificial Neural Network (ANN / MLP)
  - Random Forest Classifier

**Evaluation:** Accuracy score and classification report on held-out test set.

## 📌 Part 2 — Clustering: Customer Segmentation

**Goal:** Divide customers into five economic categories (very rich, rich, average, poor, very poor) using unsupervised learning.

**Approach:**
- Data preprocessing: label encoding, min-max scaling
- Parameter tuning for each algorithm
- Three clustering algorithms implemented and compared:
  - **K-Means** — optimal K selected via Silhouette Score
  - **DBSCAN** — eps and min_samples tuned via grid search
  - **Mean Shift** — bandwidth estimated automatically
- Cluster quality evaluated using Silhouette Score
- Results visualized with scatter plots

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models and preprocessing |
| Matplotlib & Seaborn | Data visualization |
