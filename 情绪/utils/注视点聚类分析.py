import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\眼动\lht.csv"
data = pd.read_csv(file_path)

# Update gaze position column names
x_col = 'bino_eye_gaze_position_x'  # Gaze X coordinate
y_col = 'bino_eye_gaze_position_y'  # Gaze Y coordinate
trigger_col = 'trigger'  # Trigger column to identify image changes

# Filter valid gaze points
valid_data = data[(data[x_col] > 0) & (data[y_col] > 0)]

# Identify image segments using the trigger column
image_segments = valid_data[valid_data[trigger_col] == 1].index.tolist()
image_segments.append(len(valid_data))  # Add the end of the dataset
image_data = []

# Split the data into segments for each image
for i in range(len(image_segments) - 1):
    segment = valid_data.iloc[image_segments[i]:image_segments[i + 1]]
    image_data.append(segment)

# Perform clustering and visualization for each image
for i, segment in enumerate(image_data):
    # Prepare data for clustering
    gaze_points = segment[[x_col, y_col]].values

    # Perform K-means clustering
    n_clusters = 6  # Number of clusters (adjustable)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(gaze_points)

    # Get cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the gaze points with cluster labels
    plt.figure(figsize=(12, 8))
    for cluster_id in range(n_clusters):
        cluster_points = gaze_points[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}', alpha=0.5)

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

    # Configure plot
    plt.title(f'Gaze Points Clustering Analysis for Image {i + 1}', fontsize=16)
    plt.xlabel('Screen X Position', fontsize=14)
    plt.ylabel('Screen Y Position', fontsize=14)
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.gca().invert_yaxis()  # Match screen coordinate system
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
