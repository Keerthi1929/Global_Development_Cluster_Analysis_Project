🌍 Global Development Clustering Analysis

This is a Streamlit web app for exploring and analyzing world development indicators. The project clusters countries based on economic, social, demographic, environmental, and technological features using Gaussian Mixture Models (GMM) and PCA.

Features

Data Preprocessing

Handles missing values with median imputation.

Optional outlier capping using the IQR method.

Feature Selection

Select which indicators to include in clustering.

Clustering & Metrics

GMM applied on PCA-transformed data.

Cluster evaluation using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.

Visualizations

PCA scatter plots for cluster separation.

Cluster distribution bar charts.

Feature importance plots from PCA components.

Country-Level Analysis

Filter and inspect countries by cluster.

Download Results

Export clustered data as CSV for further analysis.

Model Info

Display GMM components, covariance type, convergence status, and PCA explained variance.

Tech Stack

Python Libraries: streamlit, pandas, numpy, scikit-learn, joblib, matplotlib, seaborn

Machine Learning: Gaussian Mixture Model (GMM), PCA

Visualization: Matplotlib, Seaborn, Streamlit charts

Usage

Clone the repository.

Place the data files (clustered_data.csv, feature_names.json) and models (gaussian_mixture_model.joblib, scaler.joblib, pca.joblib) in the project directory.

Run the app:

streamlit run streamlit_app.py


Use the sidebar to:

Select features for analysis

Apply filters (countries, outlier capping)

Choose PCA axes for visualization

Insights

Groups countries with similar development patterns.

Highlights which features contribute most to cluster formation.

Helps policymakers, researchers, and analysts identify trends and benchmark countries.
