import cv2
import os

# Folder containing your images
folder_path = 'C:/Users/Toshiba_2/Desktop/acts/annotated images'

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Iterate through the image files
for image_file in image_files:
    # Load the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    if image is not None:
        # Normalize the image
        normalized_image = image.astype('float32') / 255.0
        
        # Save the normalized image back to the folder
        cv2.imwrite(image_path, (normalized_image * 255).astype('uint8'))
    else:
        print(f"Error loading {image_file}")

print("Normalization complete.")
