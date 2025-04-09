import pandas as pd
import numpy as np

# Load the CSV file
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\LHT.csv"
data = pd.read_csv(file_path)

# Gaze position columns
x_col = 'bino_eye_gaze_position_x'  # Gaze X coordinate
y_col = 'bino_eye_gaze_position_y'  # Gaze Y coordinate
trigger_col = 'trigger'  # Trigger column to identify image changes

# Define saccade threshold (in pixels)
saccade_threshold = 50  # Distance in pixels

# Identify image segments using the trigger column
image_segments = data[data[trigger_col] == 1].index.tolist()
image_segments.append(len(data))  # Add the end of the dataset
image_data = []

# Split the data into segments for each image
for i in range(len(image_segments) - 1):
    segment = data.iloc[image_segments[i]:image_segments[i + 1]]
    image_data.append(segment)

# Function to calculate saccades for a segment
def calculate_saccades(segment, x_col, y_col, threshold):
    saccades = 0
    # Iterate through the gaze points
    for i in range(1, len(segment)):
        x1, y1 = segment.iloc[i - 1][x_col], segment.iloc[i - 1][y_col]
        x2, y2 = segment.iloc[i][x_col], segment.iloc[i][y_col]
        # Calculate Euclidean distance
        if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > threshold:
            saccades += 1
    return saccades

# Initialize a list to store saccade counts for each image
saccade_counts = []

# Analyze each image's saccade count
for i, segment in enumerate(image_data):
    saccade_count = calculate_saccades(segment, x_col, y_col, saccade_threshold)
    saccade_counts.append(saccade_count)
    print(f"Image {i + 1} Saccade Count: {saccade_count}")

# Calculate overall saccades
total_saccades = sum(saccade_counts)
print(f"\nTotal Saccades Across All Images: {total_saccades}")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, len(saccade_counts) + 1), saccade_counts, color='skyblue', edgecolor='black')

# Add labels above the bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{int(bar.get_height())}", 
             ha='center', fontsize=12)

# Configure the plot
plt.title("Saccade Counts for Each Image", fontsize=16, fontweight='bold')
plt.xlabel("Image Number", fontsize=14)
plt.ylabel("Saccade Count", fontsize=14)
plt.xticks(range(1, len(saccade_counts) + 1), [f"Image {i}" for i in range(1, len(saccade_counts) + 1)], fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
