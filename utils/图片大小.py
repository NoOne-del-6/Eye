from PIL import Image
import os

def resize_image(input_path, output_path, size=(1920, 1080)):
    """
    Resize an image to the specified size (default is 1980x1080).
    :param input_path: str, path to the input image.
    :param output_path: str, path to save the resized image.
    :param size: tuple, target size as (width, height).
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert image to RGB mode if it is RGBA
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            # Resize the image
            resized_img = img.resize(size, Image.LANCZOS)

            # Save the resized image to the output path
            resized_img.save(output_path)

            print(f"Image resized successfully and saved to {output_path}")
    except Exception as e:
        print(f"Error resizing image: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Input image path
    input_image_path = r"C:\Users\Lhtooo\Desktop\6.jpg"  # Replace with your image file

    # Output image path
    output_image_path = r"C:\Users\Lhtooo\Desktop\output.jpg"  # Replace with your desired output file

    # Call the resize function
    resize_image(input_image_path, output_image_path)
