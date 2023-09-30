import cv2
import os

# Folder containing your images with inconsistent color formats
input_folder = 'C:/Users/Toshiba_2/Desktop/acts/taijiquan8'

# Folder to save images with a consistent color format (e.g., RGB)
output_folder = 'C:/Users/Toshiba_2/Desktop/acts/color_format'

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        # Convert the image to the RGB color format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save the RGB image to the output folder
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, rgb_image)
    else:
        print(f"Error loading {image_file}")

print("Color format conversion complete.")
