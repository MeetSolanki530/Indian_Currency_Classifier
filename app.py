import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Indian Currency Detection",
    page_icon = ":brain:",
    initial_sidebar_state = 'auto'
)


# Load the saved model
model = load_model(r'Model/best_currency_model.h5')
width,height = 150,150

# Define the category labels
categories = ['10 Rupees Note', '100 Rupees Note', '20 Rupees Note', '200 Rupees Note', '2000 Rupees Note', '50 Rupees Note', '500 Rupees Note']

st.title('Indian Notes Currency Classifier')


with st.sidebar:
        st.subheader("Below are sample images With result")
        st.title("100 Rupees Note")
        st.image(r'Test/image.webp')
        st.title("2000 Rupees Note")
        st.image(r'Test/note.jpeg')
        st.subheader("Below are sample video With result")
        st.title("2000 Rupee Note")
        st.video(r'Test/Video.webm')

def preprocess_image(image):
 # Load and preprocess the image
        img = load_img(image, target_size=(150, 150))  # Load image and resize to 150x150 pixels
        img_array = img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch size of 1
        img_array /= 255.0  # Rescale pixel values to [0, 1]
        return img_array



def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    predictions = model.predict(frames)

    # Aggregate predictions over frames (e.g., averaging or voting)
    aggregated_prediction = np.mean(predictions, axis=0)  # You can use other aggregation techniques based on your requirements
    predicted_label = np.argmax(aggregated_prediction)
    rounded_predictions = np.round(predictions, 2)
    predicted_class = np.argmax(predictions)
    confidence = rounded_predictions[0][predicted_class]
    st.write(f"Predicted Category: {categories[predicted_class]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    return categories[predicted_label],"Confidence: {confidence * 100:.2f}%"


def main():
    # Streamlit app
    st.title("Indian Currency Detection")
    st.write("Upload an image")
    upload_option = st.radio("Choose upload option:", ("Image", "Video"))
    if upload_option == "Image":
        image = st.file_uploader('Enter Image:- ',type=['jpg','jpeg','png','webp'])
        if st.button('Predict'):
            if image is not None:
                st.image(image,caption='Uploaded Image',use_column_width=True)
                image = preprocess_image(image)
                
                # Predict using the model
                prediction = model.predict(image)
                rounded_predictions = np.round(prediction, 2)
                predicted_class = np.argmax(prediction)
                confidence = rounded_predictions[0][predicted_class]

            st.write(f"Predicted Category: {categories[predicted_class]}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
    
    elif upload_option == "Video":
        video = st.file_uploader("Upload a video...", type=["mp4","webm","avi","mkv","wmv","flv"])
        if st.button('Predict'):
            if video is not None:
                video_path = 'temp_video.mp4'
                with open(video_path, "wb") as f:
                    f.write(video.getbuffer())

                st.video(video)
                prediction = predict_video(video_path)
                

if __name__ == "__main__":
    main()









        







    
