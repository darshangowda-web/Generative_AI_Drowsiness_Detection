import openai
import requests
from datetime import datetime
from pygame import mixer
import time
import os
import speech_recognition as sr  # Import speech recognition
import random

# API keys for OpenAI, ElevenLabs, and Google API
openai.api_key = "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"
ELEVEN_LABS_API_KEY = "sk_c19de80a320b3b9a0132f3b695ceb2115d504bcaf002ea67"
GOOGLE_API_KEY = "AIzaSyCQ3HMbev1tg8G5wVBvXjc8hDv22x3IHto"  # Google Geolocation API key

# Initialize pygame mixer for audio playback

mixer.init()

# Initialize recognizer for speech-to-text
recognizer = sr.Recognizer()

# Function to call ChatGPT API for generating a response
def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful, funny, and friendly assistant for drivers."},
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
                results = []
                for place in places[:5]:  # Get the first 5 results
                    place_name = place['name']
                    place_address = place['vicinity']
                    place_lat = place['geometry']['location']['lat']
                    place_lon = place['geometry']['location']['lng']
                    map_link = f"https://www.google.com/maps?q={place_lat},{place_lon}"
                    results.append(f"{place_name}, {place_address}. Map: {map_link}")
                return results
            else:
                return ["No nearby gas stations found."]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return ["Error fetching data from Google Places API."]
    except Exception as e:
        print(f"Exception occurred: {e}")
        return ["An error occurred while searching for gas stations."]

# Function to get current location using Google Maps Geolocation API
def get_current_location_google():
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + GOOGLE_API_KEY
    
    payload = {
        # Provide Wi-Fi access points or keep empty to rely on IP-based location
        "considerIp": "true"  # Use this to indicate the API can use the device's IP for location
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
            print(f"Error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None, None


# Function to listen for speech input
def listen_for_input():
    with sr.Microphone() as source:
        print("Listening for input... Say something!")
        
        # Adjust recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            # Listen for a specific amount of time or until silence
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # Adjust timeout and time limit
            print("Recognizing...")
            user_input = recognizer.recognize_google(audio)
            print(f"User: {user_input}")
            return user_input
        except sr.WaitTimeoutError:
            print("Listening timed out. Please speak up!")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""


# Main function for interaction
def driver_assistant():
    # Fetch the current location using Google Maps Geolocation API
    latitude, longitude = get_current_location_google()

    if latitude and longitude:
        print(f"Current Location - Latitude: {latitude}, Longitude: {longitude}")
    else:
        print("Could not fetch the location.")
        return  # Exit if location is not available

    print("Assistant is running...")

    while True:
        # Listen for driver's verbal input
        user_input = listen_for_input()

        if not user_input:
            continue  # Skip the rest if no input was recognized

        # Handle gas station or pit stop request
        if "gas station" in user_input.lower() or "pit stop" in user_input.lower():
            print("Searching for nearby gas stations...")
            results = find_nearby_gas_station_google(latitude, longitude)  # This returns a list of strings
            
            if results:
                # Show all available options in the terminal but only give the nearest one verbally
                print("All available gas stations near you:")
                for idx, result in enumerate(results[:5]):  # Print top 5 results
                    print(f"{idx + 1}. {result}")

                # Choose the nearest one (first one)
                nearest_station = results[0]
                ai_response = f"\nThe nearest gas station to you is: {nearest_station}"

            else:
                ai_response = "No nearby gas stations found."


        elif user_input.lower() == "exit":
            print("Exiting assistant.")
            break
        else:
            # General ChatGPT interaction with a funny twist
            ai_response = generate_text(user_input)

            # Add some humor based on the input
            if "hello" in user_input.lower():
                ai_response = random.choice([
                    "Hello there, traveler! Ready for an adventure?",
                    "Well, hello, my favorite driver! How's the road today?",
                    "Hey! How’s it going? Let me know if you need some tunes or info!"
                ])
            elif "tired" in user_input.lower():
                ai_response = random.choice([
                    "Tired? You better not fall asleep, or I might start singing!",
                    "I got your back! How about a coffee break or a pit stop?",
                    "Don't worry, I'll keep you awake... with my excellent jokes!"
                ])
            elif "thank you" in user_input.lower():
                ai_response = "You're very welcome! Always happy to help. Let's keep this road trip going!"

        print(f"AI: {ai_response}")

        # Convert response to speech
        audio_content = generate_audio(ai_response)
        if audio_content:
            save_and_play_audio(audio_content)


# Run the assistant
driver_assistant()