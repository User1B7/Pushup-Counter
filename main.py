"""
This script uses MediaPipe Pose to detect pushup movements in a live camera feed and counts the number of pushups performed. It also displays a real-time visualization of the pushup count and creates a plot showing the pushup count over time.

The key components of the script are as follows:

1. Initialization: It imports necessary libraries, initializes MediaPipe Pose, and sets up variables and thresholds.

2. Main Function: The main function captures frames from the camera feed, processes them to detect poses using MediaPipe, counts pushups based on pose landmarks, and displays real-time information on the frame.

3. Counting Pushups: The count_pushups function determines if a pushup movement is detected based on the angle between shoulder, elbow, and wrist joints. If the arms are below a downward angle threshold, it increments the pushup count.

4. Drawing Landmarks: The draw_landmarks function overlays pose landmarks on the camera frame.

5. Creating Pushup Count Plot: After the camera feed ends, the script generates a plot showing the pushup count over time.

6. Angle Calculation: The get_angle function calculates the angle between three given points using vector operations and returns the angle in degrees.

7. Elapsed Time Calculation: The get_elapsed_time function calculates the elapsed time since the start of pushup counting.

"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open default camera
cap = cv2.VideoCapture(0)

# Create window for camera feed
cv2.namedWindow("Pushup Counter Camera", cv2.WINDOW_NORMAL)

# Initialize variables
pushup_count = 0
last_pose_was_down = False
pushup_times = []
pushup_values = []
start_time = None

# Threshold for detecting the downward angle of the arms
DOWNWARD_ANGLE_THRESHOLD = 150  # Angle (in degrees) indicating the start of a pushup movement

def main():
    """
    Main function to capture frames, detect pushup movements, and display real-time information.
    """
    global start_time, pushup_count, pushup_times, pushup_values
    start_time = datetime.datetime.now()

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detections
            results = pose.process(rgb_frame)

            # Count pushups
            count_pushups(results.pose_landmarks)

            # Draw landmarks on the frame
            frame_with_landmarks = frame.copy()
            draw_landmarks(frame_with_landmarks, results.pose_landmarks)

            # Display pushup count and time on the frame
            overlay_text = f"Pushups: {pushup_count}  Time: {get_elapsed_time()}"
            cv2.putText(frame_with_landmarks, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with landmarks
            cv2.imshow("Pushup Counter Camera", frame_with_landmarks)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Create and display the pushup count plot
        create_pushup_count_plot(pushup_times, pushup_values)

def count_pushups(pose_landmarks):
    """
    Function to count pushups based on the angle of the arms.
    """
    global pushup_count, last_pose_was_down

    if pose_landmarks is not None and pose_landmarks.landmark:
        # Calculate the angle between shoulder, elbow, and wrist
        left_shoulder = get_angle(pose_landmarks.landmark, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
        right_shoulder = get_angle(pose_landmarks.landmark, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)

        # Check if the arms are downward
        if left_shoulder < DOWNWARD_ANGLE_THRESHOLD and right_shoulder < DOWNWARD_ANGLE_THRESHOLD:
            if not last_pose_was_down:
                pushup_count += 1
                pushup_times.append((datetime.datetime.now() - start_time).total_seconds())
                pushup_values.append(pushup_count)
            last_pose_was_down = True
        else:
            last_pose_was_down = False

def get_angle(landmarks, joint1, joint2, joint3):
    """
    Function to calculate the angle between three given points.
    """
    joint1_coords = np.array([landmarks[joint1].x, landmarks[joint1].y])
    joint2_coords = np.array([landmarks[joint2].x, landmarks[joint2].y])
    joint3_coords = np.array([landmarks[joint3].x, landmarks[joint3].y])

    vector1 = joint1_coords - joint2_coords
    vector2 = joint3_coords - joint2_coords

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    cos_theta = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return np.degrees(angle_rad)

def draw_landmarks(frame, landmarks):
    """
    Function to draw landmarks on the frame.
    """
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

def create_pushup_count_plot(times, values):
    """
    Function to create and display the pushup count plot.
    """
    plt.plot(times, values, marker='o')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pushup Count')
    plt.title('Pushup Count over Time')
    plt.grid(True)  # Show a grid in the plot
    plt.show()

def get_elapsed_time():
    """
    Function to calculate the elapsed time since the start of pushup counting.
    """
    global start_time
    elapsed_time = datetime.datetime.now() - start_time
    return str(elapsed_time)

if __name__ == "__main__":
    main()
