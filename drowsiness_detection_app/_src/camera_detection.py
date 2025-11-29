# camera_detection.py
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
from pygame import mixer

# Initialize pygame mixer for sound alerts
mixer.init()

# Load shape predictor and camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\Darshan\Desktop\drowsiness_detection_app\shape_predictor_68_face_landmarks.dat')

# Landmark indices for left eye, right eye, and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Thresholds
ear_threshold = 0.25
closed_eyes_duration = 1.5
yawn_threshold = 35
yawn_duration = 2

# Alert sound
alert_file = r"C:\Users\Darshan\Desktop\drowsiness_detection_app\static\music.wav"

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate the lip distance (for yawning detection)
def cal_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    return dist.euclidean(np.mean(top_lip, axis=0), np.mean(low_lip, axis=0))

# Function to play alert sound
def play_alert():
    mixer.music.load(alert_file)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)
    mixer.music.unload()

# Function to detect drowsiness based on eye and mouth movement
def detect_drowsiness_and_yawning(cap):
    eyes_closed_start_time = None
    yawn_start_time = None
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            lip_distance = cal_yawn(shape)

            # Add drowsiness alert logic
            if ear < ear_threshold:
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                elif time.time() - eyes_closed_start_time >= closed_eyes_duration:
                    play_alert()

            # Add yawning alert logic
            if lip_distance > yawn_threshold:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= yawn_duration:
                    play_alert()

        # Show the video feed
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
