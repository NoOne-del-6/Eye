from PIL import Image
import matplotlib.pyplot as plt

# Load images
# image_paths = [
#     r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\logistic regression_confusion_matrix.png",
#     r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\rf_confusion_matrix.png",
#     r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\svc_confusion_matrix.png",
#     r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\xgboost_confusion_matrix.png"
# ]

image_paths = [
    r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\logistic regression_roc_curve.png",
    r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\rf_roc_curve.png",
    r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\svc_roc_curve.png",
    r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\xgboost_roc_curve.png"
]

images = [Image.open(image_path) for image_path in image_paths]

# Create a new image to fit 2x2 grid
width, height = images[0].size
grid_width = width * 2
grid_height = height * 2

# Create a new blank image with the correct size
grid_image = Image.new('RGB', (grid_width, grid_height))

# Paste each image into the appropriate position
grid_image.paste(images[0], (0, 0))
grid_image.paste(images[1], (width, 0))
grid_image.paste(images[2], (0, height))
grid_image.paste(images[3], (width, height))

# Show the final combined image
plt.figure(figsize=(10, 10))
plt.imshow(grid_image)
plt.axis('off')  # Hide axes
# plt.savefig(r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\combined_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.savefig(r"C:\Users\Lhtooo\Desktop\生理信号分析\code\results\combined_roc.png", dpi=300, bbox_inches='tight')
plt.close()
