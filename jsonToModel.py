import os
import json
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define the folder containing your images
image_folder = 'C:/Users/Toshiba_2/Desktop/acts/color_format'
output_json_file = 'annotations.json'

annotations = []

for image_filename in os.listdir(image_folder):
    if image_filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_filename)
        
        # Read and process the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform pose estimation on the image
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Extract pose landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z  # If 3D landmarks are available
                })

            # Create a JSON annotation for the image
            annotation = {
                "image_path": image_path,
                "landmarks": landmarks
            }

            annotations.append(annotation)

# Write the list of annotations to a JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(annotations, json_file, indent=2)

print(f"Annotations saved to {output_json_file}")
