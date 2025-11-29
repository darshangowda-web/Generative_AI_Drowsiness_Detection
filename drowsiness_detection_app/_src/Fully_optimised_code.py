import openai
import requests
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
from pygame import mixer
from keras.models import load_model
import speech_recognition as sr
import time
import os
from datetime import datetime
#from map import find_nearby_gas_stations, get_current_location
# API keys for OpenAI and Eleven Labs
openai.api_key = "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"
ELEVEN_LABS_API_KEY = "sk_c19de80a320b3b9a0132f3b695ceb2115d504bcaf002ea67"

# Initialize pygame mixer for sound alerts
mixer.init()
alert_file = r"C:\Users\Darshan\Desktop\drowsiness_detection_app\static\music.wav"

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\Darshan\Desktop\drowsiness_detection_app\shape_predictor_68_face_landmarks.dat')

# Load the pre-trained yawning detection model
yawn_model = load_model(r'C:\Users\Darshan\Desktop\drowsiness_detection_app\yawn_detection_model.keras')

# Landmark indices for left eye, right eye, and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Cooldown period after each alert (in seconds)

alert_cooldown = 10  # seconds
last_alert_time = 0  # Track the time of the last alert

# Define constants for thresholds
ear_threshold = 0.25  # Eye Aspect Ratio threshold
yawn_threshold = 35  # Yawn threshold (lip distance)
closed_eyes_duration = 2  # Seconds the eyes must be closed before alert
yawn_duration = 1  # Seconds of yawning before alert

# Initialize timers for closed eyes and yawning detection
eyes_closed_start_time = None
yawn_start_time = None

# Function to record and transcribe user speech using SpeechRecognition
def record_speech(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (speak now)")
        try:
            audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            print("Processing your input...")
            user_input = recognizer.recognize_google(audio)
            return user_input
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            return ""
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate the lip distance (used for yawning detection)
def cal_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    return dist.euclidean(np.mean(top_lip, axis=0), np.mean(low_lip, axis=0))

# Function to save and play audio response using pygame, with unique filenames
def save_and_play_audio(content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_audio_{timestamp}.mp3"
    
    with open(filename, "wb") as f:
        f.write(content)
    
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)
    
    mixer.music.unload()
    os.remove(filename)

# Generate AI response using OpenAI API
def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "I'm Ivy, a helpful voice assistant for a drowsy driver."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Convert text to audio using Eleven Labs API
def generate_audio(text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    headers = {"xi-api-key": ELEVEN_LABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0, "similarity_boost": 0}}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.content
    else:
        print("Error with Eleven Labs API:", response.status_code)
        return None

# Play alert sound with handling to reload each time
def play_alert():
    mixer.music.load(alert_file)
    mixer.music.play()
    print("Alert sound playing.")
    while mixer.music.get_busy():
        time.sleep(0.1)
    mixer.music.unload()

# Function to save conversation history into a text file


def save_conversation_to_file(conversation_history):
    # Generate a unique filename based on the current time
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = f"conversation_{timestamp}.txt"

    # Open the file with UTF-8 encoding to handle non-ASCII characters (e.g., Tamil)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for line in conversation_history:
                f.write(line + "\n")
        print(f"Conversation saved successfully as {filename}.")
    except Exception as e:
        print(f"Error saving conversation: {e}")



conversation_history = []
# Main loop for drowsiness detection and interaction
cap = cv2.VideoCapture(0)
frame_counter = 0
consec_frames = 20

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

        # Draw contours with distinct colors for eyes and mouth
        cv2.polylines(frame, [np.int32(leftEye)], isClosed=True, color=(0, 255, 0), thickness=1)  # Green contour for left eye
        cv2.polylines(frame, [np.int32(rightEye)], isClosed=True, color=(0, 255, 0), thickness=1)  # Blue contour for right eye
        cv2.polylines(frame, [np.int32(shape[mStart:mEnd])], isClosed=True, color=(0, 0, 255), thickness=1)  # Red contour for mouth

        # Add text alerts for drowsiness and yawning
        if ear < ear_threshold:
            cv2.putText(frame, "Drowsiness Alert!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()  # Start the timer when eyes are closed
            elif time.time() - eyes_closed_start_time >= closed_eyes_duration:
                play_alert()
                ai_prompt = "You seem drowsy, are you okay?"
                ai_text_response = generate_text(ai_prompt)
                conversation_history.append(f"AI: {ai_text_response}")
                print(f"AI Response: {ai_text_response}")  # Print AI response
                audio_response = generate_audio(ai_text_response)
                if audio_response:
                    save_and_play_audio(audio_response)
                
                # Start the conversation loop
                user_response = record_speech()  # Take input from the user
                print(f"User Response: {user_response}")  # Print user response
                conversation_history.append(f"User: {user_response}")  # Append user response to the history
                while True:
                    if user_response.lower() == "stop":
                        print("Conversation ended by user.")
                        save_conversation_to_file(conversation_history)  # Save conversation to file
                        eyes_closed_start_time = None  # Reset after conversation
                        break  # Exit the conversation loop
                    else:
                        ai_text_response = generate_text(f"User said: {user_response}")
                        conversation_history.append(f"AI: {ai_text_response}")
                        print(f"AI Response: {ai_text_response}")
                        audio_response = generate_audio(ai_text_response)
                        if audio_response:
                            save_and_play_audio(audio_response)
                        user_response = record_speech()  # Take input from the user
                        print(f"User Response: {user_response}")  # Print user response
                        conversation_history.append(f"User: {user_response}")  # Append user response to the history

                # Return to drowsiness detection after conversation
                eyes_closed_start_time = None  # Reset after alert

        else:
            eyes_closed_start_time = None  # Reset if eyes are not closed

# For yawning detection, you could feed the lip distance or some other features into the model
# After detecting the mouth shape (or any other relevant features), use the model to predict yawning
        if lip_distance > yawn_threshold:
            cv2.putText(frame, "Yawning Alert!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            try:
                if yawn_start_time is None:
                    yawn_start_time = time.time()  # Start the timer when yawning is detected
                elif time.time() - yawn_start_time >= yawn_duration:
                    play_alert()
                    ai_prompt = "I noticed you yawning. Do you need a break?"
                    ai_text_response = generate_text(ai_prompt)
                    conversation_history.append(f"AI: {ai_text_response}")
                    print(f"AI Response: {ai_text_response}")  # Print AI response
                    audio_response = generate_audio(ai_text_response)
                    if audio_response:
                        save_and_play_audio(audio_response)
                    
                    # Start the conversation loop
                    #user_response = record_speech()  # Take input from the user
                    #print(f"User Response: {user_response}")  # Print user response
                    #conversation_history.append(f"User: {user_response}")  # Append user response to the history

                    while True:
                        if user_response.lower() == "stop":
                            print("Conversation ended by user.")
                            yawn_start_time = None  # Reset after conversation
                            break  # Exit the conversation loop
                        else:
                            ai_text_response = generate_text(f"User said: {user_response}")
                            print(f"AI Response: {ai_text_response}")
                            audio_response = generate_audio(ai_text_response)
                            if audio_response:
                                save_and_play_audio(audio_response)
                            #user_response = record_speech()  # Take input from the user
                            #print(f"User Response: {user_response}")  # Print user response
                            #conversation_history.append(f"User: {user_response}")  # Append user response to the history

                    # Return to yawning detection after conversation
                    yawn_start_time = None  # Reset after alert

            except Exception as e:
                #print(f"Error during yawning detection: {e}")
                
                # If there's an error, fallback to using model-based prediction
                try:
                    # Create a dummy image for model prediction
                    dummy_image = np.ones((80, 80, 3), dtype=np.float32) * lip_distance
                    dummy_image = dummy_image / 255.0  # Normalize
                    dummy_image = np.expand_dims(dummy_image, axis=0)  # Shape (1, 80, 80, 3)

                    prediction = yawn_model.predict(dummy_image)  # Use model for prediction
                    if prediction > 0.5:  # If the model predicts yawning
                        play_alert()
                        ai_prompt = "I noticed you yawning. Do you need a break?"
                        ai_text_response = generate_text(ai_prompt)
                        conversation_history.append(f"AI: {ai_text_response}")
                        print(f"AI Response: {ai_text_response}")
                        audio_response = generate_audio(ai_text_response)
                        if audio_response:
                            save_and_play_audio(audio_response)
                    else:
                        #print("Model did not predict yawning")
                        cv2.putText(frame, "No Yawn Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Model prediction error: {e}")
                    print("Falling back to manual yawning check...")

                    # Fallback action after model prediction failure
                    play_alert()
                    ai_prompt = "I noticed you yawning. Do you need a break?"
                    ai_text_response = generate_text(ai_prompt)
                    print(f"AI Response: {ai_text_response}")
                    audio_response = generate_audio(ai_text_response)
                    if audio_response:
                        save_and_play_audio(audio_response)
                        

        else:
            yawn_start_time = None  # Reset if no yawning is detected
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

