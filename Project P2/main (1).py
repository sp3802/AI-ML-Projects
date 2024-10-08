import streamlit as st
import requests
import json
import uuid
import azure.cognitiveservices.speech as speechsdk

# Set up the Azure Custom Vision API credentials
prediction_key = "670ab71352cc48489a92fe5842413ff8"  # Your Prediction Key
endpoint = "https://smartplantassistant-prediction.cognitiveservices.azure.com/"
project_id = "22f675aa-ab23-4e05-afdc-14b9d657e371"  # Your Project ID
publish_iteration_name = "Iteration2"  # Your published iteration name

# Streamlit UI code
st.title("Plant Disease Classification")

# Upload Image Section
st.header("Upload Plant Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Make prediction when an image is uploaded
if uploaded_file is not None:
    # Read the image file
    image_data = uploaded_file.read()

    # Define headers and URL for prediction
    headers = {
        'Prediction-Key': prediction_key,
        'Content-Type': 'application/octet-stream'
    }
    url = f"{endpoint}customvision/v3.0/Prediction/{project_id}/classify/iterations/{publish_iteration_name}/image"

    # Make the prediction
    response = requests.post(url, headers=headers, data=image_data)

    # Check for successful response
    if response.status_code == 200:
        result = response.json()
        # Display the result
        st.success("Prediction successful!")
        predictions = result.get('predictions', [])

        if predictions:
            # Find the prediction with the highest probability
            best_prediction = max(predictions, key=lambda x: x['probability'])
            tag_name = best_prediction['tagName']
            probability = best_prediction['probability']
            st.write(f"**Predicted Class**: {tag_name} (Probability: {probability:.2%})")
        else:
            st.warning("No predictions found.")
    else:
        st.error(f"Error: {response.status_code}, {response.text}")


# Azure OpenAI settings
openai_api_key = "61ad35e6fa9d46759ae53e5b4b02f354"  # Your OpenAI key
openai_endpoint = "https://smartplantopenai.openai.azure.com/"
openai_deployment_name = "openaigpt35turbo"  # Deployed model name

# Azure Translator settings
translator_key = 'd71d5ad9793c494ba91af058942f73df'  # Your Translator API key
translator_endpoint = 'https://api.cognitive.microsofttranslator.com/'
translator_location = 'eastus'  # Your Translator region

# Azure Speech Service settings
speech_key = "2d8ecb53901f475684918912bb47172e"  # Your Speech API key
speech_region = "eastus"  # Your Speech service region

# Function to generate disease diagnosis using OpenAI
def generate_disease_suggestion(prompt):
    data = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": None
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key,
    }
    response = requests.post(
        f"{openai_endpoint}/openai/deployments/{openai_deployment_name}/completions?api-version=2023-03-15-preview", 
        headers=headers, json=data
    )
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["text"].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to translate text using Azure Translator API
def translate_text(text, target_languages):
    path = '/translate?api-version=3.0'
    params = f'&to={",".join(target_languages)}'
    constructed_url = translator_endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': translator_location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    response = requests.post(constructed_url, headers=headers, json=body)
    result = response.json()

    translations = {translation['to']: translation['text'] for translation in result[0]['translations']}
    return translations

# Function to convert text to speech using Azure Speech Service
def text_to_speech(text, language):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_language = language  # Set language for speech synthesis
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text(text)

# Function to translate and speak
def translate_and_speak(prediction, languages):
    language_map = {
        'es': 'es-ES',  # Spanish
        'fr': 'fr-FR',  # French
        'te': 'te-IN',  # Telugu
        'hi': 'hi-IN',  # Hindi
        'de': 'de-DE',  # German
        'en': 'en-US'  # English
    }

    translations = translate_text(prediction, languages)
    for lang, translated_text in translations.items():
        st.write(f"Translated Diagnosis in {lang.upper()}: {translated_text}")
        text_to_speech(translated_text, language_map[lang])



# Plant Disease Diagnosis Section
st.header("Plant Disease Diagnosis")
plant_symptoms = st.text_input("Enter plant symptoms for diagnosis:", "Black Rot")
if st.button("Get Diagnosis"):
    prompt = f"Provide a possible plant disease diagnosis for the following symptoms: {plant_symptoms}. This is related to agricultural use."
    diagnosis = generate_disease_suggestion(prompt)
    st.session_state.diagnosis = diagnosis  # Store diagnosis in session state
    st.write(f"Diagnosis: {diagnosis}")

# Display stored diagnosis if available
if 'diagnosis' in st.session_state:
    st.write(f"Diagnosis (from session state): {st.session_state.diagnosis}")

# Translate and Speak Diagnosis Section
st.header("Translate and Speak Diagnosis")
languages = st.multiselect("Select languages to translate to", ['en', 'es', 'fr', 'te', 'hi', 'de'])
if st.button("Translate and Speak"):
    if 'diagnosis' in st.session_state:
        translate_and_speak(st.session_state.diagnosis, languages)
    else:
        st.write("Please get a diagnosis first.")

