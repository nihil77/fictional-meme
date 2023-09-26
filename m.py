import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to determine the pose label
def determine_pose_label(results):
    # Get the landmarks for the Nose and Left Hip
    nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

    # Calculate the vertical distance between Nose and Left Hip
    distance = nose_landmark.y - left_hip_landmark.y

    # Define a threshold to distinguish between poses (customize as needed)
    threshold = 0.1  # Adjust this threshold based on your application

    # Determine the pose label
    if distance > threshold:
        return "Standing"
    else:
        return "Sitting"

# Function to detect and draw keypoints on the frame
def detect_pose(frame):
    # Convert the frame to RGB format (MediaPipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation on the frame
    results = pose.process(frame_rgb)

    # Check if landmarks are found
    if results.pose_landmarks is not None:
        height, width, _ = frame.shape
        
        # Draw skeleton lines connecting landmarks
        for connection in mp_pose.POSE_CONNECTIONS:
            start_point = connection[0]
            end_point = connection[1]
            start_x, start_y = int(results.pose_landmarks.landmark[start_point].x * width), int(results.pose_landmarks.landmark[start_point].y * height)
            end_x, end_y = int(results.pose_landmarks.landmark[end_point].x * width), int(results.pose_landmarks.landmark[end_point].y * height)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Display landmark confidence scores
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            x, y = int(landmark.x * width), int(landmark.y * height)
            confidence = landmark.visibility  # Confidence score
            cv2.putText(frame, f"{idx}: {confidence:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Add pose label
        pose_label = determine_pose_label(results)
        cv2.putText(frame, f"Pose: {pose_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

# Open a video capture stream from the webcam (use video source 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Detect and draw pose landmarks on the frame
    frame_with_landmarks = detect_pose(frame)

    # Display the frame with landmarks in real-time
    cv2.imshow("Pose Landmarker", frame_with_landmarks)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
