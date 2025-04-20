from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

file_path = "/Users/jaredog/Downloads/git code/SC4021-Info-retrieval/ClassificationNew/appended_dataset.csv"  # Replace with your actual Excel file path
data = pd.read_csv(file_path)

# Step 1: Vectorize the cleaned text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(data['cleaned_body'])

# Step 2: Add subjectivity and polarity as features
X_features = data[['subjectivity', 'polarity']].values

# Combine the text features and sentiment features
from scipy.sparse import hstack
X_combined = hstack([X_text, X_features])

# Step 4: Convert 'entities' column into binary features (one-hot encoding)
mlb = MultiLabelBinarizer()
entity_features = mlb.fit_transform(data['entities'].apply(eval))  # Convert string lists into actual lists

# Combine entity features with the other features
X_combined_with_entities = hstack([X_combined, entity_features])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
X_combined_with_entities_pca = pca.fit_transform(X_combined_with_entities.toarray())

# Step 5: Apply KMeans clustering
kmeans_with_entities = KMeans(n_clusters=6, random_state=42)
kmeans_with_entities.fit(X_combined_with_entities_pca)

# Add cluster labels to the dataframe
predicted_labels = kmeans_with_entities.labels_
predicted_labels = predicted_labels.astype(int)

# Step 6: Map clusters to sentiment labels
# Create a mapping based on the majority sentiment in each cluster
cluster_sentiment_mapping = {}

for cluster in range(kmeans_with_entities.n_clusters):
    # Find the rows that belong to this cluster
    cluster_rows = data[predicted_labels == cluster]
    # Majority sentiment in the cluster
    majority_sentiment = cluster_rows['label'].mode()[0]
    # Map this cluster to the majority sentiment
    cluster_sentiment_mapping[cluster] = majority_sentiment

# Map predicted labels to sentiment labels
mapped_sentiment_labels = [cluster_sentiment_mapping[label] for label in predicted_labels]

# Step 7: Get the true labels for the remaining rows
true_labels = data['label']

# Step 8: Evaluate clustering performance using Precision, Recall, F1-Score
precision = precision_score(true_labels, mapped_sentiment_labels, average='weighted', zero_division=1)  # Handling zero divisions
recall = recall_score(true_labels, mapped_sentiment_labels, average='weighted', zero_division=1)
f1 = f1_score(true_labels, mapped_sentiment_labels, average='weighted', zero_division=1)
accuracy = accuracy_score(true_labels, mapped_sentiment_labels)

# Print evaluation metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# 1. Calculate Centroid Distances (Euclidean distance between centroids)
centroids = kmeans_with_entities.cluster_centers_
inter_cluster_distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
print("Inter-cluster centroid distances:\n", inter_cluster_distances)

# 2. Calculate Silhouette Score
silhouette_avg = silhouette_score(X_combined_with_entities_pca, predicted_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# 3. Calculate Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(X_combined_with_entities_pca, predicted_labels)
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# 4. Calculate Average Intra-cluster Similarity (average distance within clusters)
# Intra-cluster distance (average distance within each cluster)
intra_cluster_similarity = []
for cluster_id in range(kmeans_with_entities.n_clusters):
    cluster_points = X_combined_with_entities_pca[predicted_labels == cluster_id]
    cluster_center = centroids[cluster_id]
    intra_cluster_similarity.append(np.mean(np.linalg.norm(cluster_points - cluster_center, axis=1)))

average_intra_cluster_similarity = np.mean(intra_cluster_similarity)
print(f"Average Intra-cluster Similarity: {average_intra_cluster_similarity:.4f}")