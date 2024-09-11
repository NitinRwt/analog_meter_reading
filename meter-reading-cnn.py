import gradio as gr
import cv2
import numpy as np
import pytesseract
from PIL import Image
import tensorflow as tf  # or whatever framework you used for your CNN

def load_cnn_model(model_path):
    # Load your saved CNN model
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model at the start of your script
cnn_model = load_cnn_model('CNN_Analog-Readout.h5')

def process_digital_meter(image):
    # Existing digital meter processing code...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((1, 1), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    text = pytesseract.image_to_string(threshold, config='--psm 8')
    try:
        value = float(text)
    except ValueError:
        value = None
    return value

def process_analog_meter_cnn(image):
    # Preprocess the image as required by your CNN model
    processed_image = preprocess_for_cnn(image)
    
    # Use the model to predict the meter reading
    prediction = cnn_model.predict(processed_image)
    
    # Process the prediction as needed (e.g., convert to a readable value)
    meter_reading = process_prediction(prediction)
    
    return meter_reading

def preprocess_for_cnn(image):
    # Implement preprocessing steps required by your CNN model
    # This might include resizing, normalization, etc.
    # Example:
    resized = cv2.resize(image, (224, 224))  # Adjust size as per your model's input
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=0)
    return expanded

def process_prediction(prediction):
    # Process the raw prediction from your CNN model
    # This depends on how your model outputs the prediction
    # Example:
    meter_reading = prediction[0][0]  # Adjust based on your model's output format
    return meter_reading

def read_meter(image, meter_type):
    if meter_type == "Digital":
        reading = process_digital_meter(image)
    else:
        reading = process_analog_meter_cnn(image)
    
    if reading is not None:
        return f"The meter reading is: {reading:.2f}"
    else:
        return "Failed to extract reading. Please try again with a clearer image."

iface = gr.Interface(
    fn=read_meter,
    inputs=[
        gr.Image(type="numpy", label="Upload Meter Image"),
        gr.Radio(["Digital", "Analog"], label="Meter Type")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Meter Reading Application",
    description="Upload an image of a digital or analog meter to get the reading."
)

iface.launch()
