import os
import streamlit as st
from google.cloud import aiplatform
import base64
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from difflib import SequenceMatcher

# Function to calculate similarity between two strings
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to get image description
def get_image_description_gemini(model, uploaded_file):
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    image_url = f"data:image/png;base64,{encoded_image}"

    prompt = f"Describe the following image: {image_url}"

    response = model.chat([InputOutputTextPair(Input=prompt)])
    
    description = response.output
    return description

# Streamlit app layout
st.title("Image Relevance to News Text using Gemini 1.5 Pro")
st.write("Upload a news text and images to find out which images are relevant to the text.")

# Authenticate and initialize Vertex AI SDK
PROJECT_ID = "your-google-cloud-project-id"
LOCATION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Load the Gemini 1.5 Pro model
model = ChatModel.from_pretrained("chat-gemini-1.5-pro")

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
            description = get_image_description_gemini(model, uploaded_file)
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
