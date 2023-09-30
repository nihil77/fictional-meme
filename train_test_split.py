from sklearn.model_selection import train_test_split
import os

# Folder containing your preprocessed and annotated images
data_folder = 'C:/Users/Toshiba_2/Desktop/acts/color_format'

# List all image files in the data folder
image_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Split the dataset into training, validation, and test sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

# Now you have lists of file paths for each set
print(f"Number of training examples: {len(train_files)}")
print(f"Number of validation examples: {len(val_files)}")
print(f"Number of test examples: {len(test_files)}")
