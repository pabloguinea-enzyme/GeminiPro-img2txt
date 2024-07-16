import os
import streamlit as st
import base64
from difflib import SequenceMatcher
from google.cloud import aiplatform
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

# Function to calculate similarity between two strings
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to get image description using Gemini
def get_image_description_gemini(chat, uploaded_file):
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    image_url = f"data:image/png;base64,{encoded_image}"

    prompt = f"Describe the following image: {image_url}"
    description = get_chat_response(chat, prompt)
    return description

# Function to get chat response
def get_chat_response(chat, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)

# Streamlit app layout
st.title("Image Relevance to News Text using Gemini 1.5 Pro")
st.write("Upload a news text and images to find out which images are relevant to the text.")

# Textbox for API key
api_key = st.text_input("Enter your API Key", type="password")

if not api_key:
    st.error("Please provide a valid API Key.")
    st.stop()

# Set API Key in environment variable
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize Vertex AI SDK
project_id = "bright-aloe-429610-d4"
location = "us-central1"

try:
    aiplatform.init(project=project_id, location=location)
except Exception as e:
    st.error(f"Error initializing Vertex AI: {e}")
    st.stop()

# Load the Gemini 1.5 Flash model
model_id = "chat-gemini-1.5-pro-001"
try:
    model = ChatModel.from_pretrained(model_id)
    chat = model.start_chat()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Text area for news text input
news_text = st.text_area("Enter the news text")

# Upload multiple images button
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if news_text and uploaded_files:
    # Store descriptions and similarity scores
    image_relevancies = []

    for uploaded_file in uploaded_files:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")

            # Get the image description
            description = get_image_description_gemini(chat, uploaded_file)
            st.write(description)

            # Calculate similarity between news text and image description
            similarity = calculate_similarity(news_text, description)
            image_relevancies.append({
                'similarity': similarity,
                'image': uploaded_file,
                'description': description
            })
        except Exception as e:
            st.error(f"Error: {e}")

    # Sort images by relevance
    image_relevancies = sorted(image_relevancies, key=lambda x: x['similarity'], reverse=True)

    st.write("Relevant Images:")
    for item in image_relevancies:
        st.image(item['image'], caption=f"Description: {item['description']} (Similarity: {item['similarity']:.2f})", use_column_width=True)
else:
    st.error("Please provide a valid news text and upload images.")
