import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Load the CSV file
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\LHT.csv"
data = pd.read_csv(file_path)

# Update gaze position column names
x_col = 'bino_eye_gaze_position_x'  # Gaze X coordinate
y_col = 'bino_eye_gaze_position_y'  # Gaze Y coordinate
trigger_col = 'trigger'  # Trigger column to identify image changes

# Define AOI (Area of Interest) boundaries for a single six-grid layout
aoi_bounds = {
    "anger": [0, 640, 0, 540],       # Top-left
    "disgust": [640, 1280, 0, 540],  # Top-center
    "fear": [1280, 1920, 0, 540],    # Top-right
    "joy": [0, 640, 540, 1080],      # Bottom-left
    "sadness": [640, 1280, 540, 1080], # Bottom-center
    "surprise": [1280, 1920, 540, 1080] # Bottom-right
}

# Identify image segments using the trigger column
image_segments = data[data[trigger_col] == 1].index.tolist()
image_segments.append(len(data))  # Add the end of the dataset
image_data = []

# Split the data into segments for each image
for i in range(len(image_segments) - 1):
    segment = data.iloc[image_segments[i]:image_segments[i + 1]]
    image_data.append(segment)

# Initialize a list to store percentages for each image
all_image_percentages = []

# Analyze each image's gaze distribution
for i, segment in enumerate(image_data):
    # Initialize gaze counts for this image
    emotion_gaze_counts = {emotion: 0 for emotion in aoi_bounds.keys()}

    # Count gaze points for each AOI in the segment
    for x, y in zip(segment[x_col], segment[y_col]):
        for emotion, bounds in aoi_bounds.items():
            x_min, x_max, y_min, y_max = bounds
            if x_min <= x <= x_max and y_min <= y <= y_max:
                emotion_gaze_counts[emotion] += 1

    # Calculate percentages
    total_gaze_points = sum(emotion_gaze_counts.values())
    if total_gaze_points > 0:
        emotion_percentages = {emotion: (count / total_gaze_points) * 100 for emotion, count in emotion_gaze_counts.items()}
    else:
        emotion_percentages = {emotion: 0 for emotion in aoi_bounds.keys()}
    all_image_percentages.append(emotion_percentages)

# Define radar chart settings
labels = list(aoi_bounds.keys())  # Emotions (labels)
num_vars = len(labels)

# Convert data into a format suitable for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Create subplots for each image
plt.figure(figsize=(12, 12))

# Colors for each image, using soft gradients
colors = ['#4C72B0', '#55A868', '#C44E52', '#FFB947']  # Soft, distinct colors for each image
labels_image = [f"Image {i + 1}" for i in range(len(all_image_percentages))]

# Plot combined radar chart for all images
ax = plt.subplot(111, polar=True)
for idx, emotion_percentages in enumerate(all_image_percentages):
    values = list(emotion_percentages.values())
    values += values[:1]  # Repeat the first value to close the circle
    angle_shifted = angles + [angles[0]]  # Shift to complete the circular chart
    
    # Plot the line for each image with transparency
    ax.plot(angle_shifted, values, color=colors[idx], linewidth=3, label=labels_image[idx], alpha=0.9)
    ax.fill(angle_shifted, values, color=colors[idx], alpha=0.15)  # Filling the area with lower transparency

# Set the labels and title
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=16, fontweight='bold', color='darkslategray')  # Set emotion labels
ax.set_yticklabels([])  # Remove radial ticks
ax.set_title("Combined Gaze Distribution for All Images", size=24, weight='bold', color='darkslategray')

# Customize grid and background
ax.grid(color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_facecolor('#f0f0f0')  # Light gray background for the plot area

# Add a legend for clarity
ax.legend(loc='upper right', fontsize=14, frameon=False, title="Images", title_fontsize=16)

# Fine-tune the layout
plt.tight_layout()
plt.show()










# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the CSV file
# file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\data\眼动\lht.csv"
# data = pd.read_csv(file_path)

# # Update gaze position column names
# x_col = 'bino_eye_gaze_position_x'  # Gaze X coordinate
# y_col = 'bino_eye_gaze_position_y'  # Gaze Y coordinate
# trigger_col = 'trigger'  # Trigger column to identify image changes

# # Define AOI (Area of Interest) boundaries for a single six-grid layout
# aoi_bounds = {
#     "anger": [0, 640, 0, 540],       # Top-left
#     "disgust": [640, 1280, 0, 540],  # Top-center
#     "fear": [1280, 1920, 0, 540],    # Top-right
#     "joy": [0, 640, 540, 1080],      # Bottom-left
#     "sadness": [640, 1280, 540, 1080], # Bottom-center
#     "surprise": [1280, 1920, 540, 1080] # Bottom-right
# }

# # Identify image segments using the trigger column
# image_segments = data[data[trigger_col] == 1].index.tolist()
# image_segments.append(len(data))  # Add the end of the dataset
# image_data = []

# # Split the data into segments for each image
# for i in range(len(image_segments) - 1):
#     segment = data.iloc[image_segments[i]:image_segments[i + 1]]
#     image_data.append(segment)

# # Initialize a list to store percentages for each image
# all_image_percentages = []

# # Analyze each image's gaze distribution
# for i, segment in enumerate(image_data):
#     # Initialize gaze counts for this image
#     emotion_gaze_counts = {emotion: 0 for emotion in aoi_bounds.keys()}

#     # Count gaze points for each AOI in the segment
#     for x, y in zip(segment[x_col], segment[y_col]):
#         for emotion, bounds in aoi_bounds.items():
#             x_min, x_max, y_min, y_max = bounds
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 emotion_gaze_counts[emotion] += 1

#     # Calculate percentages
#     total_gaze_points = sum(emotion_gaze_counts.values())
#     if total_gaze_points > 0:
#         emotion_percentages = {emotion: (count / total_gaze_points) * 100 for emotion, count in emotion_gaze_counts.items()}
#     else:
#         emotion_percentages = {emotion: 0 for emotion in aoi_bounds.keys()}
#     all_image_percentages.append(emotion_percentages)

# # Convert to DataFrame for easy plotting
# df_percentages = pd.DataFrame(all_image_percentages)

# # Plot the stacked bar chart
# plt.figure(figsize=(16, 10))
# ax = df_percentages.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='coolwarm', width=0.8)

# # Add labels on top of each bar
# for p in ax.patches:
#     height = p.get_height()
#     if height > 0:
#         ax.text(p.get_x() + p.get_width() / 2., p.get_y() + height / 2, f'{height:.1f}%', ha="center", va="center", fontsize=10, color="white", fontweight="bold")

# # Configure plot appearance
# plt.xlabel("Image", fontsize=18, fontweight='bold', color='darkblue')
# plt.ylabel("Percentage of Gaze Points (%)", fontsize=18, fontweight='bold', color='darkblue')
# plt.title("Gaze Distribution Across Different Emotions for All Images", fontsize=20, fontweight='bold', color='darkred')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(title="Emotions", fontsize=14)
# plt.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.7, color='gray')
# plt.tight_layout()
# plt.show()