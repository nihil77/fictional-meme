import cv2
import os

# Folder containing your original images
input_folder = 'C:/Users/Toshiba_2/Desktop/acts/annotated images'

# Folder to save resized images
output_folder = 'C:/Users/Toshiba_2/Desktop/acts/taijiquan8'

# Target resolution
target_width, target_height = 400, 400

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        # Resize the image
        resized_image = cv2.resize(image, (target_width, target_height))
        
        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, resized_image)
    else:
        print(f"Error loading {image_file}")

print("Image resizing complete.")
