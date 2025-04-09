import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\眼动\lht.csv"
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
total_gaze_counts = {emotion: 0 for emotion in aoi_bounds.keys()}  # Overall gaze counts

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

    # Add to total counts
    for emotion, count in emotion_gaze_counts.items():
        total_gaze_counts[emotion] += count

    # Calculate percentages for the current image
    total_gaze_points = sum(emotion_gaze_counts.values())
    if total_gaze_points > 0:
        emotion_percentages = {emotion: (count / total_gaze_points) * 100 for emotion, count in emotion_gaze_counts.items()}
    else:
        emotion_percentages = {emotion: 0 for emotion in aoi_bounds.keys()}  # No gaze points in the image

    all_image_percentages.append(emotion_percentages)

    # Print percentages for this image
    print(f"Image {i + 1} Gaze Distribution Percentages:")
    for emotion, percentage in emotion_percentages.items():
        print(f"  {emotion.capitalize()}: {percentage:.2f}%")

# Calculate overall percentages across all images
total_gaze_points_overall = sum(total_gaze_counts.values())

# Avoid division by zero error
if total_gaze_points_overall > 0:
    overall_percentages = {emotion: (count / total_gaze_points_overall) * 100 for emotion, count in total_gaze_counts.items()}
else:
    overall_percentages = {emotion: 0 for emotion in total_gaze_counts.keys()}  # No gaze points overall

# Print overall percentages
print("\nOverall Gaze Distribution Percentages:")
for emotion, percentage in overall_percentages.items():
    print(f"{emotion.capitalize()}: {percentage:.2f}%")

# Plot bar charts for all images
plt.figure(figsize=(16, 10))
colors = plt.cm.plasma(np.linspace(0, 1, len(all_image_percentages)))  # Use a modern color map with gradient

for i, emotion_percentages in enumerate(all_image_percentages):
    bars = plt.bar(
        [f"{emotion} (Img {i+1})" for emotion in emotion_percentages.keys()],
        emotion_percentages.values(),
        label=f"Image {i + 1}",
        alpha=0.85,
        color=colors[i],
        edgecolor='black', linewidth=1.2
    )

    # Add percentage labels on top of each bar (simulate shadow effect by adding two layers)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='white',
                 verticalalignment='bottom', horizontalalignment='center')

        # Simulate shadow by adding black text with small offset
        plt.text(bar.get_x() + bar.get_width() / 2.0 + 0.05, height + 0.5, f"{height:.2f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black',
                 verticalalignment='bottom', horizontalalignment='center')

# Configure plot appearance for individual images
plt.xlabel("Emotion (per Image)", fontsize=18, fontweight='bold', color='darkblue')
plt.ylabel("Percentage of Gaze Points (%)", fontsize=18, fontweight='bold', color='darkblue')
plt.title("Gaze Distribution Across Different Emotions for All Images", fontsize=20, fontweight='bold', color='darkred')
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.7, color='gray')
plt.legend(title="Images", fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# Plot overall percentages
plt.figure(figsize=(14, 8))
overall_bars = plt.bar(overall_percentages.keys(), overall_percentages.values(), color='deepskyblue', edgecolor='black', linewidth=1.5)

# Add percentage labels on top of each bar (simulate shadow effect by adding two layers)
for bar in overall_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}%",
             ha='center', va='bottom', fontsize=14, fontweight='bold', color='white', verticalalignment='bottom', horizontalalignment='center')

    # Simulate shadow by adding black text with small offset
    plt.text(bar.get_x() + bar.get_width() / 2.0 + 0.05, height + 0.5, f"{height:.2f}%",
             ha='center', va='bottom', fontsize=14, fontweight='bold', color='black', verticalalignment='bottom', horizontalalignment='center')

# Configure plot appearance for overall percentages
plt.xlabel("Emotion", fontsize=18, fontweight='bold', color='darkblue')
plt.ylabel("Percentage of Gaze Points (%)", fontsize=18, fontweight='bold', color='darkblue')
plt.title("Overall Gaze Distribution Across Different Emotions", fontsize=20, fontweight='bold', color='darkred')
plt.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.7, color='gray')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
