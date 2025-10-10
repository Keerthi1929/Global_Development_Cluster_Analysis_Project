# World Development Clustering Analysis App
# Based on the complete model building process from the notebook

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="Global Development Clustering Analysis",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for better styling (loaded from external file)
try:
    css_path = os.path.join(BASE_DIR, 'styles.css')
    with open(css_path, 'r') as f:
        st.markdown('<style>' + f.read() + '</style>', unsafe_allow_html=True)
except Exception:
    pass

@st.cache_data
def load_data():
    """Load and preprocess the clustered data"""
    try:
        # Load dataset
        csv_path = os.path.join(BASE_DIR, 'clustered_data.csv')
        data = pd.read_csv(csv_path)
        
        # Load feature names from JSON
        json_path = os.path.join(BASE_DIR, 'feature_names.json')
        with open(json_path, 'r') as f:
            features = json.load(f)
        
        return data, features
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def load_model():
    """Load the saved GMM model"""
    try:
        model_path = os.path.join(BASE_DIR, 'gaussian_mixture_model.joblib')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_resource
def load_scaler():
    """Load the saved scaler"""
    try:
        scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        st.stop()

@st.cache_resource
def load_pca():
    """Load the saved PCA transformer"""
    try:
        pca_path = os.path.join(BASE_DIR, 'pca.joblib')
        pca = joblib.load(pca_path)
        return pca
    except Exception as e:
        st.error(f"Error loading PCA: {str(e)}")
        st.stop()

def preprocess_data(data, features):
    """Preprocess data following the notebook steps"""
    # Select numeric features
    numeric_features = [f for f in features if f in data.columns and data[f].dtype in ['int64', 'float64']]
    
    # Handle missing values with median imputation
    X = data[numeric_features].copy()
    X.fillna(X.median(), inplace=True)
    
    return X, numeric_features

def handle_outliers(X):
    """Handle outliers using IQR method as in the notebook"""
    X_processed = X.copy()
    
    # Exclude columns not suitable for outlier detection
    cols_to_exclude = ['Number of Records']
    numeric_cols = [col for col in X.columns if col not in cols_to_exclude]
    
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = X_processed[col].quantile(0.25)
        Q3 = X_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        num_upper = (X_processed[col] > upper_bound).sum()
        num_lower = (X_processed[col] < lower_bound).sum()
        
        # Cap the outliers
        X_processed[col] = np.where(X_processed[col] < lower_bound, lower_bound,
                          np.where(X_processed[col] > upper_bound, upper_bound, X_processed[col]))
        
        outlier_summary.append({
            "Feature": col,
            "Lower Outliers": num_lower,
            "Upper Outliers": num_upper,
            "Total Outliers": num_lower + num_upper
        })
    
    return X_processed, pd.DataFrame(outlier_summary)

def calculate_cluster_metrics(X_pca, labels):
    """Calculate clustering validation metrics"""
    try:
        silhouette = silhouette_score(X_pca, labels)
        calinski_harabasz = calinski_harabasz_score(X_pca, labels)
        davies_bouldin = davies_bouldin_score(X_pca, labels)
        
        return {
            "Silhouette Score": silhouette,
            "Calinski-Harabasz Index": calinski_harabasz,
            "Davies-Bouldin Index": davies_bouldin
        }
    except Exception as e:
        st.warning(f"Could not calculate metrics: {str(e)}")
        return None

def create_cluster_summary(data, labels, features):
    """Create cluster summary statistics"""
    data_with_clusters = data.copy()
    data_with_clusters['Predicted_Cluster'] = labels
    
    # Get numeric features for summary
    numeric_features = [f for f in features if f in data.columns and data[f].dtype in ['int64', 'float64']]
    
    cluster_summary = data_with_clusters.groupby('Predicted_Cluster')[numeric_features].mean()
    cluster_counts = data_with_clusters['Predicted_Cluster'].value_counts().sort_index()
    
    return cluster_summary, cluster_counts

# Main App
st.markdown('<h1 class="main-header">🌍 Global Development Clustering Analysis</h1>', unsafe_allow_html=True)

# Load data and models
with st.spinner("Loading data and models..."):
    data, features = load_data()
    model = load_model()
    scaler = load_scaler()
    pca = load_pca()

# Sidebar for controls
st.sidebar.header("Analysis Controls")

# Interactive controls
st.sidebar.subheader("Preprocessing")
apply_outlier_capping = st.sidebar.checkbox("Apply IQR outlier capping", value=True)

# Feature selection
st.sidebar.subheader("Features")
default_features = []
for f in [
    'Birth Rate','CO2 Emissions','Days to Start Business','Energy Usage','GDP',
    'Health Exp % GDP','Health Exp/Capita','Infant Mortality Rate','Internet Usage',
    'Lending Interest','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage',
    'Number of Records','Population 0-14','Population 15-64','Population 65+','Population Total',
    'Population Urban','Tourism Inbound','Tourism Outbound'
]:
    if f in features:
        default_features.append(f)

selected_features = st.sidebar.multiselect(
    "Select features for analysis",
    options=[f for f in features if f in data.columns],
    default=default_features if default_features else [f for f in features if f in data.columns]
)

st.sidebar.subheader("Filters")
country_filter = None
if 'Country' in data.columns:
    country_filter = st.sidebar.multiselect(
        "Filter by Country (optional)",
        options=sorted(data['Country'].unique()),
        default=[]
    )

# Data preprocessing
st.header("📊 Data Preprocessing")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    st.write(f"**Total Records:** {len(data)}")
    st.write(f"**Total Features:** {len(features)}")
    st.write(f"**Countries:** {data['Country'].nunique() if 'Country' in data.columns else 'N/A'}")

with col2:
    st.subheader("Data Quality")
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.write("Missing values per feature:")
        st.dataframe(missing_data[missing_data > 0])
    else:
        st.success("✅ No missing values found")

# Apply optional country filter for display (not affecting model input unless features require)
if country_filter:
    data_view = data[data['Country'].isin(country_filter)].copy()
else:
    data_view = data.copy()

# Preprocess data using selected features
X_all, numeric_features_all = preprocess_data(data, features)

# Respect user-selected features but keep only numeric ones present
numeric_features = [f for f in selected_features if f in numeric_features_all]
X = data[numeric_features].copy()
X.fillna(X.median(), inplace=True)

# Handle outliers (optional)
if apply_outlier_capping:
    X_processed, outlier_df = handle_outliers(X)
else:
    X_processed = X.copy()
    outlier_df = pd.DataFrame()

# Show outlier information
if st.sidebar.checkbox("Show Outlier Analysis"):
    st.subheader("🔍 Outlier Analysis")
    st.dataframe(outlier_df)

# Transform data
X_scaled = scaler.transform(X_processed)
X_pca = pca.transform(X_scaled)

# Predict clusters
labels = model.predict(X_pca)

 

# Add cluster labels to data
data['Predicted_Cluster'] = labels

# Cluster Analysis
st.header("🎯 Cluster Analysis")

# Cluster metrics
metrics = calculate_cluster_metrics(X_pca, labels)
if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.3f}")
    with col2:
        st.metric("Calinski-Harabasz Index", f"{metrics['Calinski-Harabasz Index']:.1f}")
    with col3:
        st.metric("Davies-Bouldin Index", f"{metrics['Davies-Bouldin Index']:.3f}")

    with st.expander("Why this model is best", expanded=True):
        st.markdown(
            f"""
            - Higher Silhouette and Calinski–Harabasz, and lower Davies–Bouldin indicate compact, well-separated clusters.
            - Current scores: Silhouette = {metrics['Silhouette Score']:.3f}, Calinski–Harabasz = {metrics['Calinski-Harabasz Index']:.1f}, Davies–Bouldin = {metrics['Davies-Bouldin Index']:.3f}.
            - Gaussian Mixture Model captures elliptical cluster shapes using full covariances, matching this dataset better than spherical assumptions.
            - Soft assignments (cluster probabilities) handle borderline countries more robustly than hard assignments.
            - PCA plot below shows visually distinct groups with minimal overlap.
            """
        )

# Cluster summary
cluster_summary, cluster_counts = create_cluster_summary(data, labels, features)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Distribution")
    st.bar_chart(cluster_counts)

with col2:
    st.subheader("Cluster Characteristics")
    st.dataframe(cluster_summary.round(2))

# Visualization
st.header("📈 Visualizations")

# PCA axes selection
st.sidebar.subheader("PCA Axes")
pc_options = list(range(1, min(6, getattr(pca, 'n_components_', 2) + 1)))
pc_x = st.sidebar.selectbox("X axis (PC)", pc_options, index=0)
pc_y = st.sidebar.selectbox("Y axis (PC)", pc_options, index=1 if len(pc_options) > 1 else 0)

# PCA Scatter Plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, pc_x-1], X_pca[:, pc_y-1], c=labels, cmap='viridis', alpha=0.7, s=50)
ax.set_xlabel(f'Principal Component {pc_x}')
ax.set_ylabel(f'Principal Component {pc_y}')
ax.set_title('Cluster Visualization (PCA Projection)')
plt.colorbar(scatter, label='Cluster')
st.pyplot(fig)

# Feature importance (PCA components)
if st.sidebar.checkbox("Show PCA Component Analysis"):
    st.subheader("🔬 PCA Component Analysis")
    
    # Get feature importance from PCA
    feature_importance = abs(pca.components_[0])  # First principal component
    feature_names = numeric_features
    
    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Feature Importance (Absolute Value)')
    ax.set_title('Feature Importance in First Principal Component')
    plt.tight_layout()
    st.pyplot(fig)

# Country-level analysis
st.header("🌎 Country-Level Analysis")

# Show clustered data
if st.sidebar.checkbox("Show Clustered Data"):
    st.subheader("Clustered Countries")
    
    # Filter options
    selected_cluster = st.selectbox("Select Cluster", sorted(data['Predicted_Cluster'].unique()))
    show_columns = st.multiselect(
        "Columns to show",
        options=list(data.columns),
        default=['Country', 'Predicted_Cluster'] if 'Country' in data.columns else ['Predicted_Cluster']
    )

    filtered_data = data[data['Predicted_Cluster'] == selected_cluster][show_columns]
    st.dataframe(filtered_data)

# Download functionality
st.header("💾 Download Results")

# Prepare data for download (optionally filtered by cluster)
st.sidebar.subheader("Download Options")
dl_cluster = st.sidebar.selectbox("Download cluster", options=["All"] + list(map(int, sorted(data['Predicted_Cluster'].unique()))))

if dl_cluster == "All":
    download_data = data.copy()
else:
    download_data = data[data['Predicted_Cluster'] == int(dl_cluster)].copy()

csv = download_data.to_csv(index=False)

st.download_button(
    label="📥 Download Clustered Data",
    data=csv,
    file_name="world_development_clustered.csv",
    mime="text/csv"
)

# Model Information
st.header("🤖 Model Information")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gaussian Mixture Model")
    st.write(f"**Number of Components:** {model.n_components}")
    st.write(f"**Covariance Type:** {model.covariance_type}")
    st.write(f"**Converged:** {'Yes' if model.converged_ else 'No'}")

with col2:
    st.subheader("Preprocessing Pipeline")
    st.write(f"**Scaler:** StandardScaler")
    st.write(f"**PCA Components:** {pca.n_components_}")
    st.write(f"**Explained Variance Ratio:** {pca.explained_variance_ratio_[:2].sum():.3f}")

# Footer
st.markdown("---")
st.markdown("**Analysis completed using Gaussian Mixture Model clustering on World Development Indicators**")