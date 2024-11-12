# frontend/app.py
import streamlit as st
import requests
from PIL import Image

# Set up the backend API URL
API_URL = 'http://34.27.81.23:5000/predict'

# Page layout settings
st.set_page_config(page_title="Enhanced Decision Tree Prediction Demo", layout="centered")

# Display a header and image for aesthetics
st.title("🌳 Enhanced Decision Tree Classifier")
st.write("Predict the class of a sample using a trained Decision Tree model. Adjust the input values and click 'Predict'.")

# Show an optional introductory image for user engagement
# image = Image.open('sample_image.jpg')  # Optional image to visually enhance UI
# st.image(image, caption="Classifying data points with machine learning", use_column_width=True)

# Improved input fields with user guidance
st.write("### Input Features")
with st.form("prediction_form"):
    feature_1 = st.number_input("Feature 1 (e.g., sepal length)", min_value=0.0, max_value=10.0, step=0.1, help="Enter a numeric value")
    feature_2 = st.number_input("Feature 2 (e.g., sepal width)", min_value=0.0, max_value=10.0, step=0.1, help="Enter a numeric value")
    feature_3 = st.number_input("Feature 3 (e.g., petal length)", min_value=0.0, max_value=10.0, step=0.1, help="Enter a numeric value")
    feature_4 = st.number_input("Feature 4 (e.g., petal width)", min_value=0.0, max_value=10.0, step=0.1, help="Enter a numeric value")

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Input validation and loading spinner for user feedback
    if any(val is None for val in [feature_1, feature_2, feature_3, feature_4]):
        st.error("Please fill in all features to get a prediction.")
    else:
        with st.spinner("Fetching prediction..."):
            try:
                data = [feature_1, feature_2, feature_3, feature_4]
                response = requests.post(API_URL, json={"data": data})
                prediction = response.json()["prediction"]
                st.success(f"The predicted class is: {prediction}")
            except requests.exceptions.RequestException:
                st.error("Error connecting to the backend. Please try again later.")
