import openai
import os
import requests

def get_user_input():
    """Get words from the user, separated by commas."""
    words = input("Enter one or multiple words (separated by commas): ").strip().split(',')
    return [word.strip() for word in words if word.strip()]

def generate_meaning_and_prompt(word):
    """Generate meanings, synonyms, and image prompts using OpenAI."""
    try:
        # Request meanings and prompt from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Provide details for the given word as specified.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Provide the following details for the word '{word}':\n"
                        "1. Simple meaning in very basic terms.\n"
                        "2. One-word synonym.\n"
                        "3. A silly way to remember the word by breaking it down.\n"
                        "4. A pop culture reference (movie, sitcom, or game).\n"
                        "5. An example sentence using the word.\n"
                        "6. An imaginative visual description for an image based on the word."
                    ),
                },
            ],
            max_tokens=200,
        )

        # Parse response line by line
        output = response['choices'][0]['message']['content'].strip().split("\n")
        
        # Map expected fields to extracted lines with defaults for robustness
        meanings = {
            "word": word,
            "intellectual_meaning": next((line[3:].strip() for line in output if line.startswith("1. ")), "N/A"),
            "synonym": next((line[3:].strip() for line in output if line.startswith("2. ")), "N/A"),
            "silly_remember": next((line[3:].strip() for line in output if line.startswith("3. ")), "N/A"),
            "reference": next((line[3:].strip() for line in output if line.startswith("4. ")), "N/A"),
            "example_sentence": next((line[3:].strip() for line in output if line.startswith("5. ")), "N/A"),
            "image_prompt": next((line[3:].strip() for line in output if line.startswith("6. ")), f"A creative and imaginative artistic representation of the word '{word}'."),
        }

        return meanings, meanings["image_prompt"]

    except Exception as e:
        print(f"Error generating meaning for {word}: {e}")
        return None, None

def create_ai_image(word, image_prompt):
    """Generate an image using OpenAI's image API based on the prompt."""
    try:
        # Fallback if image_prompt is empty
        if not image_prompt:
            image_prompt = f"A creative and imaginative artistic representation of the word '{word}'."

        response = openai.Image.create(
            prompt=image_prompt,
            n=1,
            size="1024x1024",
        )
        # Save image locally
        image_url = response['data'][0]['url']
        image_data = requests.get(image_url).content

        folder_name = "generated_images"
        os.makedirs(folder_name, exist_ok=True)

        file_path = os.path.join(folder_name, f"{word}.png")
        with open(file_path, 'wb') as image_file:
            image_file.write(image_data)

        return file_path
    except Exception as e:
        print(f"Error generating image for {word}: {e}")
        return None

def save_word_details(word_details, file_path):
    """Save the word details and image file path to a text file."""
    folder_name = "word_details"
    os.makedirs(folder_name, exist_ok=True)

    file_name = os.path.join(folder_name, f"{word_details['word']}.txt")
    with open(file_name, 'w') as file:
        file.write(f"Word: {word_details['word']}\n")
        file.write(f"Simple Meaning: {word_details['intellectual_meaning']}\n")
        file.write(f"One-word Synonym: {word_details['synonym']}\n")
        file.write(f"Silly Breakdown: {word_details['silly_remember']}\n")
        file.write(f"Pop Culture Reference: {word_details['reference']}\n")
        file.write(f"Example Sentence: {word_details['example_sentence']}\n")
        file.write(f"Image Prompt: {word_details['image_prompt']}\n")
        if file_path:
            file.write(f"Image Path: {file_path}\n")

def main():
    """Main workflow to process user input, generate details, and create images."""
    openai.api_key = ""  # Replace with your actual OpenAI API key
    words = get_user_input()

    for word in words:
        print(f"\nProcessing: {word}\n")

        # Generate meanings and image prompt
        word_details, image_prompt = generate_meaning_and_prompt(word)
        if not word_details:
            print(f"Skipping {word} due to an error.\n")
            continue

        # Display details
        print(f"Simple Meaning: {word_details['intellectual_meaning']}")
        print(f"One-word Synonym: {word_details['synonym']}")
        print(f"Silly Breakdown: {word_details['silly_remember']}")
        print(f"Pop Culture Reference: {word_details['reference']}")
        print(f"Example Sentence: {word_details['example_sentence']}")
        print(f"Image Prompt: {word_details['image_prompt']}\n")

        # Create an AI-generated image
        file_path = create_ai_image(word, image_prompt)

        # Save word details and image info
        save_word_details(word_details, file_path)

        print(f"Completed processing for: {word}\n")

if __name__ == "__main__":
    main()
