import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Read an image
image = cv2.imread('C:/Users/Toshiba_2/Desktop/acts/image/Golden Rooster Stance.jpg')

# Convert the image to RGB format (MediaPipe requires RGB images)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


results = pose.process(image_rgb)

# Check if landmarks are found
if results.pose_landmarks is not None:
    # Access and process pose landmarks as needed
    for landmark in results.pose_landmarks.landmark:
        x, y, z = landmark.x, landmark.y, landmark.z  # X, Y, and Z coordinates of the landmark
        # You can perform specific actions based on the landmark positions


# Draw landmarks on the image
for landmark in results.pose_landmarks.landmark:
    height, width, _ = image.shape
    x, y = int(landmark.x * width), int(landmark.y * height)
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at the landmark position

# Save or display the annotated image
cv2.imwrite('annotated_Golden_Rooster_Stance.jpg', image)
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
