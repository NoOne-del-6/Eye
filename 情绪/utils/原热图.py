import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"D:\working\生理信号分析\data\lht.csv"
data = pd.read_csv(file_path)

# Parameters
sampling_rate = 200  # Hz
image_duration = 15  # seconds
points_per_image = sampling_rate * image_duration  # Data points per image
num_images = 4  # Total number of images

# Verify we have enough data
total_points = num_images * points_per_image
if len(data) < total_points:
    raise ValueError("Data length is insufficient for 4 images with 15 seconds each at 200Hz.")

# Improved heatmap visualization accounting for rectangular screen dimensions
def plot_heatmap(data, title):
    plt.figure(figsize=(12, 6))  # Adjust figure size to reflect 1920x1080 aspect ratio
    
    # Extract X and Y gaze positions and filter valid values
    gaze_x = data['left_eye_gaze_position_x']
    gaze_y = data['left_eye_gaze_position_y']
    valid_mask = (gaze_x > 0) & (gaze_y > 0)  # Remove invalid gaze points
    gaze_x = gaze_x[valid_mask]
    gaze_y = gaze_y[valid_mask]
    
    # Create 2D histogram for heatmap
    heatmap, xedges, yedges = np.histogram2d(
        gaze_x, gaze_y, bins=[50, 28], range=[[0, 1920], [0, 1080]]
    )  # Adjust bins to maintain rectangular aspect ratio
    
    # Plot the heatmap with improved visuals
    ax = sns.heatmap(
        heatmap.T, 
        cmap="coolwarm", 
        cbar=True, 
        xticklabels=False, 
        yticklabels=False, 
        linewidths=0.1, 
        linecolor='gray'
    )
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Screen X Position", fontsize=14)
    plt.ylabel("Screen Y Position", fontsize=14)
    plt.gca().invert_yaxis()  # Match screen coordinate system
    plt.show()

# Plot improved heatmaps for each image segment
for i in range(num_images):
    start_idx = i * points_per_image
    end_idx = start_idx + points_per_image
    segment_data = data.iloc[start_idx:end_idx]
    plot_heatmap(segment_data, f"Improved Heatmap of Gaze Points for Image {i + 1}")
