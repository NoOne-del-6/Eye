import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the CSV file
file_path = r"D:\working\生理信号分析\data\lht.csv"
data = pd.read_csv(file_path)

# Parameters
sampling_rate = 200  # Hz
image_duration = 15  # seconds
points_per_image = sampling_rate * image_duration  # Data points per image
num_images = 4  # Total number of images

# Function to plot scan path with start and end points labeled
def plot_scan_path_with_labels(data, title):
    plt.figure(figsize=(14, 8))  # Adjust aspect ratio to 16:9

    # Extract X and Y gaze positions and filter valid values
    gaze_x = data['left_eye_gaze_position_x']
    gaze_y = data['left_eye_gaze_position_y']
    valid_mask = (gaze_x > 0) & (gaze_y > 0)  # Remove invalid gaze points
    gaze_x = gaze_x[valid_mask].reset_index(drop=True)
    gaze_y = gaze_y[valid_mask].reset_index(drop=True)

    # Plot scan path as a smooth line
    plt.plot(gaze_x, gaze_y, color='blue', linewidth=2, alpha=0.7, label='Scan Path')

    # Add fixation points
    plt.scatter(gaze_x, gaze_y, c='red', s=15, alpha=0.8, label='Fixation Points')

    # Highlight start and end points
    if not gaze_x.empty and not gaze_y.empty:
        plt.scatter(gaze_x.iloc[0], gaze_y.iloc[0], c='green', s=50, label='Start Point', edgecolor='black', zorder=5)
        plt.scatter(gaze_x.iloc[-1], gaze_y.iloc[-1], c='purple', s=50, label='End Point', edgecolor='black', zorder=5)
        plt.text(gaze_x.iloc[0], gaze_y.iloc[0], 'Start', fontsize=12, color='green', fontweight='bold', zorder=6)
        plt.text(gaze_x.iloc[-1], gaze_y.iloc[-1], 'End', fontsize=12, color='purple', fontweight='bold', zorder=6)

    # Configure plot appearance
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Screen X Position", fontsize=14)
    plt.ylabel("Screen Y Position", fontsize=14)
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.gca().invert_yaxis()  # Match screen coordinate system
    plt.legend(fontsize=12, loc='upper right', frameon=True)
    plt.grid(alpha=0.3, linestyle='--')
    plt.show()

# Plot scan path with labeled start and end points for each image segment
for i in range(num_images):
    start_idx = i * points_per_image
    end_idx = start_idx + points_per_image
    segment_data = data.iloc[start_idx:end_idx]
    plot_scan_path_with_labels(segment_data, f"Scan Path with Start and End Points for Image {i + 1}")
