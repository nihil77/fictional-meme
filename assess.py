import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant pose landmarks for Ma Bu (Horse Stance)
HIP_LANDMARK = mp_pose.PoseLandmark.LEFT_HIP
KNEE_LANDMARK = mp_pose.PoseLandmark.LEFT_KNEE
ANKLE_LANDMARK = mp_pose.PoseLandmark.LEFT_ANKLE

# Define angle thresholds for a correct Ma Bu stance (adjust as needed)
HIP_ANGLE_THRESHOLD = 160  # Degrees
KNEE_ANGLE_THRESHOLD = 160  # Degrees
ANKLE_ANGLE_THRESHOLD = 160  # Degrees

# Function to calculate the angle between three landmarks
def calculate_angle(a, b, c):
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ac = math.dist(a, c)
    radians = math.acos((ab**2 + bc**2 - ac**2) / (2 * ab * bc))
    degrees = math.degrees(radians)
    return degrees

# Function to determine the correctness of Ma Bu stance
def determine_ma_bu_stance(frame):
    # Convert the frame to RGB format (MediaPipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation on the frame
    results = pose.process(frame_rgb)

    # Check if landmarks are found
    if results.pose_landmarks is not None:
        # Get landmark positions
        landmarks = results.pose_landmarks.landmark

        # Get landmark positions for relevant body parts
        hip = (landmarks[HIP_LANDMARK.value].x, landmarks[HIP_LANDMARK.value].y)
        knee = (landmarks[KNEE_LANDMARK.value].x, landmarks[KNEE_LANDMARK.value].y)
        ankle = (landmarks[ANKLE_LANDMARK.value].x, landmarks[ANKLE_LANDMARK.value].y)

        # Calculate angles at hip, knee, and ankle
        hip_knee_angle = calculate_angle(hip, knee, ankle)

        # Provide feedback based on angle thresholds
        feedback = "Correct Ma Bu Stance"
        if hip_knee_angle < HIP_ANGLE_THRESHOLD:
            feedback = "Adjust Hip Angle"
        elif hip_knee_angle < KNEE_ANGLE_THRESHOLD:
            feedback = "Adjust Knee Angle"
        elif hip_knee_angle < ANKLE_ANGLE_THRESHOLD:
            feedback = "Adjust Ankle Angle"

        # Draw feedback on the frame
        cv2.putText(frame, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Open a video capture stream from the webcam (use video source 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Detect and provide feedback on Ma Bu stance
    frame_with_feedback = determine_ma_bu_stance(frame)

    # Display the frame with feedback in real-time
    cv2.imshow("Ma Bu Stance Detection", frame_with_feedback)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
