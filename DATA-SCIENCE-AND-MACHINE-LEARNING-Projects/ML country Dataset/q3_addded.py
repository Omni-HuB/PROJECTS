import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics



def load_dataset(file_path):
    """Load the data."""
    return pd.read_csv(file_path)


    # Modify the handle_missing_values function
def handle_missingvalues(data):
    data = data.replace('?', pd.NA)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)  # Filling missing values with mean
    return data


def preprocessdata(data):
    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop('num', axis=1))
    data = pd.DataFrame(scaled_data, columns=data.columns[:-1])
    data['num'] = data['num'].astype(int)
    return data

def visualizedata(data):
    sns.pairplot(data, hue='num')
    plt.title('Pairplot for the Dataset')
    plt.show()



def explore_data(data):
    """Perform exploratory data analysis (EDA)."""
    print(data.info())
    print(data.describe())
    sns.pairplot(data)
    plt.show()


def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    data = pd.read_csv(url, names=None)
    return data


def preprocess_data(data):
    """Preprocess the data by standardizing features."""
    if 'country' in data.columns:
        country_column = data['country']
        data = data.drop('country', axis=1)
    else:
        country_column = None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # If 'Country' column was present, add it back to the DataFrame
    if country_column is not None:
        scaled_df = pd.concat([country_column, scaled_df], axis=1)

    return scaled_df

def pca(scaled_data, target_variance=0.95):
    """Implement PCA and visualize results."""
    # Exclude non-numeric columns before applying PCA
    numeric_data = scaled_data.select_dtypes(include=[np.number])

    pca = PCA()
    pca_result = pca.fit_transform(numeric_data)

    # Identify the optimal number of components based on explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    optimal_components = np.argmax(cumulative_var_ratio >= target_variance) + 1

    # Visualize explained variance ratio
    plt.plot(range(1, len(explained_var_ratio) + 1), cumulative_var_ratio, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Ratio')
    plt.show()

    # Select the optimal number of components
    pca = PCA(n_components=optimal_components)
    pca_result = pca.fit_transform(numeric_data)

    # Create scatter plots for visualization
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
    plt.title('PCA Scatter Plot')
    plt.show()

    return pca_result



def kmeans(pca_result, optimal_k):
    """Apply K-Means clustering with the optimal number of clusters."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)

    # Plot clusters on a 2D scatter plot
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('K-Means Clustering Results')
    plt.show()

    return cluster_labels


def determine_optimal_k(pca_result, k_values=None):
    """Determine the optimal number of clusters using the elbow method and silhouette score."""
    if k_values is None:
        k_values = range(2, 10)

    # Elbow Method
    inertia = [KMeans(n_clusters=k, random_state=42).fit(pca_result).inertia_ for k in k_values]
    plt.plot(k_values, inertia, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # Silhouette Score
    silhouette_scores = [metrics.silhouette_score(pca_result, KMeans(n_clusters=k, random_state=42).fit_predict(pca_result)) for k in k_values]
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')
    plt.show()

    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k



def cluster_analysis(data, pca_result, cluster_labels):
    """Perform an analysis of the clusters to identify characteristics."""
    clustered_data = pd.concat([data['country'], pd.DataFrame(pca_result, columns=[f'PC{i}' for i in range(1, pca_result.shape[1] + 1)]), pd.Series(cluster_labels, name='Cluster')], axis=1)
    cluster_analysis = clustered_data.groupby('Cluster').mean()
    return cluster_analysis

if __name__ == '__main__':

    file_path = 'Country-data.csv'
    data = load_dataset(file_path)
    data.info()

    explore_data(data)
    scaled_data = preprocess_data(data)

    pca_result = pca(scaled_data)
    optimal_k = determine_optimal_k(pca_result)
    cluster_labels = kmeans(pca_result, optimal_k)
    analysis = cluster_analysis(data, pca_result, cluster_labels)

    print("printing Clustering analysis")
    print(analysis)
