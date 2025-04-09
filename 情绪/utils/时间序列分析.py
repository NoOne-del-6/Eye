import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"C:\Users\Lhtooo\Desktop\生理信号分析\code\data\眼动\zzq.csv"
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

# Function to plot combined features with validity checks and minimum point removal
def plot_combined_features(data, features, title, ylabel, feature_labels, validity_features, points_to_remove=5):
    plt.figure(figsize=(14, 8))
    for i in range(num_images):
        # Slice data for the current image
        start_idx = i * points_per_image
        end_idx = start_idx + points_per_image
        segment_data = data.iloc[start_idx:end_idx]

        
        segment_data = segment_data[(segment_data[validity_features[0]] == 1) & (segment_data[validity_features[1]] == 1)]

       
        for feature in features:
            min_idx = segment_data[feature].idxmin()
            # Ensure indices to remove are within the current segment range
            remove_indices = [idx for idx in range(max(min_idx - points_to_remove, start_idx), min(min_idx + points_to_remove + 1, end_idx)) if idx in segment_data.index]
            segment_data = segment_data.drop(index=remove_indices)

        valid_data_count = len(segment_data)
        print(f"Number of valid data points for plotting: {valid_data_count}")
        # Create a relative time axis
        timestamps_relative = (segment_data.index - start_idx) / sampling_rate  # Seconds

        # Plot the combined features for this image
        for feature, label in zip(features, feature_labels):
            plt.plot(
                timestamps_relative + i * image_duration,  # Offset each image by its starting time
                segment_data[feature], 
                label=f'Image {i + 1} - {label}', 
                alpha=0.7
            )
        
        # Add a vertical line to indicate the photo switch
        if i > 0:
            plt.axvline(x=i * image_duration, color='red', linestyle='--')

    # Add labels, title, and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Call the function with updated parameters for validity check and data cleaning
plot_combined_features(
    data, 
    ['left_eye_pupil_diameter_mm', 'right_eye_pupil_diameter_mm'], 
    'Pupil Diameter (Left and Right) Over Time', 
    'Pupil Diameter (mm)', 
    ['Left Pupil', 'Right Pupil'],
    ['left_eye_valid', 'right_eye_valid']  # Validity checks for both eyes
)
