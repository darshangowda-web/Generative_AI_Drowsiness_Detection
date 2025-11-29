import openai
import requests
from datetime import datetime
from pygame import mixer
import time
import os
import speech_recognition as sr

# API keys for OpenAI, ElevenLabs, and Google API
openai.api_key = "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"
ELEVEN_LABS_API_KEY = "sk_ac9b26877d63b40bd371274e54d2401444d6b172f33df10c"
GOOGLE_API_KEY = "AIzaSyCQ3HMbev1tg8G5wVBvXjc8hDv22x3IHto"  # Google Geolocation API key

# Initialize pygame mixer for audio playback
mixer.init()

# Function to call ChatGPT API for generating a response
def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant for drivers."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function to generate audio using ElevenLabs API
def generate_audio(text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    headers = {"xi-api-key": ELEVEN_LABS_API_KEY, "Content-Type": "application/json"}
    data = {"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error with ElevenLabs API: {response.status_code}")
        return None

# Function to save and play the audio response
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

# Function to find nearby gas stations using Google Places API
def find_nearby_gas_station_google(latitude, longitude):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius=3000&type=gas_station&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            places = response.json().get("results", [])
            if places:
                place_name = places[0]['name']
                place_address = places[0]['vicinity']
                place_lat = places[0]['geometry']['location']['lat']
                place_lon = places[0]['geometry']['location']['lng']
                map_link = f"https://www.google.com/maps?q={place_lat},{place_lon}"
                return place_name, place_address, map_link
            else:
                return "No nearby gas stations found.", None, None
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Error fetching data from Google Places API.", None, None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return "An error occurred while searching for gas stations.", None, None

# Function to get current location using Google Maps Geolocation API
def get_current_location_google():
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + GOOGLE_API_KEY
    
    payload = {
        "wifiAccessPoints": []  # If you don't have Wi-Fi access data, you can leave this empty, or you can add Wi-Fi data here
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()  # Parse the JSON response

            # Extract the latitude and longitude from the response
            latitude = data['location']['lat']
            longitude = data['location']['lng']

            return latitude, longitude
        else:
            print(f"Error: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None, None

# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't understand that."
    except sr.RequestError:
        return "Sorry, there was an issue with the speech recognition service."

# Main function for interaction
def driver_assistant():
    # Display a welcome message
    print("Welcome to your Driver Assistant! How can I help you today?")
    welcome_audio = generate_audio("Welcome to your Driver Assistant! How can I help you today?")
    if welcome_audio:
        save_and_play_audio(welcome_audio)

    # Fetch the current location using Google Maps Geolocation API
    latitude, longitude = get_current_location_google()

    if latitude and longitude:
        print(f"Current Location - Latitude: {latitude}, Longitude: {longitude}")
    else:
        print("Could not fetch the location.")
        return  # Exit if location is not available

    print("Assistant is running...")

    while True:
        # Use speech-to-text for user input
        user_input = recognize_speech()

        # Handle gas station or pit stop request
        if "gas station" in user_input.lower() or "pit stop" in user_input.lower():
            print("Searching for nearby gas stations...")
            place_name, place_address, map_link = find_nearby_gas_station_google(latitude, longitude)
            if map_link:
                ai_response = f"I found a gas station nearby: {place_name}, {place_address}. Here is the map link: {map_link}"
            else:
                ai_response = place_name
        elif "exit" in user_input.lower():
            print("Exiting assistant.")
            ai_response = "Goodbye!"
            exit_audio = generate_audio(ai_response)
            if exit_audio:
                save_and_play_audio(exit_audio)
            break
        else:
            # General ChatGPT interaction
            ai_response = generate_text(user_input)

        print(f"AI: {ai_response}")

        # Convert response to speech
        audio_content = generate_audio(ai_response)
        if audio_content:
            save_and_play_audio(audio_content)

# Run the assistant
driver_assistant()
