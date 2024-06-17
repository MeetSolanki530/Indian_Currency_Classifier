import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model(r'C:\Users\solan\Downloads\Indian Notes Currency Classifier\Model\best_currency_model.h5')

# Define the category labels
categories = ['10 Rupees Note', '100 Rupees Note', '20 Rupees Note', '200 Rupees Note', '2000 Rupees Note', '50 Rupees Note', '500 Rupees Note']

st.title('Indian Notes Currency Classifier')

st.write('Upload an Image of Currency Note')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if st.button('Get Predict'):

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Load and preprocess the image
        img = load_img(uploaded_file, target_size=(150, 150))  # Load image and resize to 150x150 pixels
        img_array = img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch size of 1
        img_array /= 255.0  # Rescale pixel values to [0, 1]
        
        # Predict the class probabilities
        predictions = model.predict(img_array)
        rounded_predictions = np.round(predictions, 2)
        predicted_class = np.argmax(predictions)
        confidence = rounded_predictions[0][predicted_class]

        st.write(f"Predicted Category: {categories[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
