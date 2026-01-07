🌍 Global Development Clustering Analysis

This project is a Streamlit web application for exploring and analyzing a Global Development (Deployment) dataset.
It clusters countries based on economic, social, demographic, environmental, and technological indicators using Gaussian Mixture Models (GMM) and Principal Component Analysis (PCA).

---
## Features
* Data Preprocessing - Handles missing values using median imputation and Optional outlier capping using the IQR method
* Feature Selection - Allows users to select development indicators for clustering
* Clustering & Evaluation Metrics - Gaussian Mixture Model (GMM) applied on PCA-transformed data
* Cluster evaluation using: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index

---
## Visualizations

* PCA scatter plots for cluster separation

* Cluster distribution bar charts

* Feature importance plots based on PCA components

## Country-Level Analysis - Filter and inspect countries by cluster

# Download Results

Export clustered country data as CSV for further analysis

---

## Tech Stack

Programming Language: Python

Libraries: Streamlit, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, Seaborn

Machine Learning: Gaussian Mixture Model (GMM), PCA

Visualization: Streamlit charts, Matplotlib, Seaborn

---

## Usage

# Clone the repository

Place the following files in the project directory:

clustered_data.csv

feature_names.json

gaussian_mixture_model.joblib

scaler.joblib

pca.joblib

# Run the application:

streamlit run streamlit_app.py

# Use the sidebar to:

Select features for clustering

Apply filters (countries, outlier capping)

Choose PCA axes for visualization

---

## Insights

Groups countries with similar global development patterns

Identifies key indicators contributing to cluster formation

Helps policymakers, researchers, and analysts compare and benchmark countries

## KMeans Cluster Visualization (PCA):
This plot shows clear separation of country clusters using KMeans after PCA dimensionality reduction.

<img width="1184" height="842" alt="Screenshot 2026-01-07 211941" src="https://github.com/user-attachments/assets/8934a654-a213-48d8-8ecd-5a16fb68c70b" />

## Clustering Algorithm Comparison:
This graph compares clustering models using silhouette score and shows KMeans performs the best.

<img width="862" height="537" alt="Screenshot 2026-01-07 211932" src="https://github.com/user-attachments/assets/936b6a9f-46ac-4a69-b57b-65193e28d276" />
