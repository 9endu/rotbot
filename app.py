import os
import numpy as np
import datetime
import re
import time
import serial
import threading
import firebase_admin
from firebase_admin import credentials, db
import joblib
import torch
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Firebase Initialization
cred = credentials.Certificate("firebase/rotbot-b300b-firebase-adminsdk-fbsvc-3b9c6a4580.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rotbot-b300b-default-rtdb.firebaseio.com/'
})

# Load ML Model, Scaler, and Label Encoder
model = joblib.load('catboost_model.pkl')  # Adjust path to your model
scaler = joblib.load('scaler.pkl')  # Adjust path to your scaler
le = joblib.load('label_encoder.pkl')  # Adjust path to your label encoder

# --- Serial Reading Thread ---
def read_serial_and_upload():
    try:
        ser = serial.Serial('COM3', 115200, timeout=2)
        print("✅ Serial port COM3 opened.")
    except Exception as e:
        print(f"❌ Serial port error: {e}")
        return

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            print("Raw Serial:", line)

            match = re.search(r'Temp: ([\d.]+)°C \| Humidity: ([\d.]+)% \| MQ-4 Sensor Value: (\d+)', line)
            if match:
                temp = float(match.group(1))
                hum = float(match.group(2))
                methane = int(match.group(3))

                now = datetime.datetime.now().isoformat()
                data = {
                    'timestamp': now,
                    'temperature': temp,
                    'humidity': hum,
                    'methane': methane
                }

                db.reference('sensor_readings').push(data)
                print("✔ Data sent to Firebase:", data)

        except Exception as e:
            print("⚠️ Error reading serial:", e)
            time.sleep(1)

# Start serial reading in a background thread
threading.Thread(target=read_serial_and_upload, daemon=True).start()

# --- Firebase Prediction ---
def predict_from_latest_data():
    ref = db.reference('sensor_readings')
    data = ref.order_by_key().limit_to_last(1).get()

    if not data:
        return None

    for key in data:
        entry = data[key]
        methane = entry['methane']
        temp = entry['temperature']
        hum = entry['humidity']

        # Preprocess
        scaled = scaler.transform([[temp, hum]])
        input_array = np.array([[np.log1p(methane)] + list(scaled[0])])  # Apply log1p for methane
        prediction = model.predict(input_array)
        predicted_class = le.inverse_transform(prediction)[0]

        return predicted_class

# --- Home Route ---
@app.route('/')
def home():
    return render_template('homepage.html')

# --- Firebase Prediction Route ---
@app.route('/firebase')
def firebase_predict():
    prediction = predict_from_latest_data()

    if prediction:
        # Fetch the latest sensor data from Firebase
        ref = db.reference('sensor_readings')
        data = ref.order_by_key().limit_to_last(1).get()
        
        if data:
            latest_data = next(iter(data.values()))  # Get the most recent data entry
            methane = latest_data['methane']
            temp = latest_data['temperature']
            hum = latest_data['humidity']

            return render_template('firebase.html', prediction=prediction, methane=methane, temperature=temp, humidity=hum)
    
    return render_template('firebase.html', error="No data available for prediction.")

# --- Custom Prediction Page Route ---
@app.route('/custom', methods=['GET', 'POST'])
def custom_predict():
    if request.method == 'POST':
        # Get JSON data from the request
        data = request.get_json()

        # Log transformation on methane and fetch other values
        methane = np.log1p(data['methane'])  # Apply log transformation
        temperature = data['temperature']
        humidity = data['humidity']

        # Preprocess
        scaled = scaler.transform([[temperature, humidity]])
        input_array = np.array([[methane] + list(scaled[0])])

        # Make prediction
        prediction = model.predict(input_array)
        predicted_class = le.inverse_transform(prediction)[0]

        # Log prediction
        print(f"Prediction: {predicted_class}")

        return jsonify({'prediction': predicted_class})

    return render_template('custom.html')

# --- Image Upload and Prediction ---
@app.route('/image_upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'GET':
        return render_template('image_upload.html')  # This template should have the upload form

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    
    if file:
        # Save the image temporarily
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Predict
        prediction = predict_shelf_life(image_path)
        return jsonify({'prediction': prediction})

    return jsonify({'error': 'No image found'}), 400


def predict_shelf_life(image_path):
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)

        # Ensure that the model is in evaluation mode
        model.eval()

        with torch.no_grad():
            # Make prediction using the model
            prediction = model(image_tensor)
            print(f"Model prediction: {prediction}")  # Debugging message

        # Placeholder logic for shelf life prediction based on hue
        hue = get_dominant_hue(image_path)
        print(f"Dominant hue: {hue}")  # Debugging message
        
        if hue >= 50 and hue <= 80:  # Example: Green hue range
            return f"Green - Shelf Life: 5 to 7 days"
        elif hue >= 25 and hue <= 50:  # Example: Yellow hue range
            return f"Yellow - Shelf Life: 3 to 5 days"
        elif hue >= 0 and hue <= 25:  # Example: Brown hue range
            return f"Brown - Shelf Life: 1 to 3 days"
        else:
            return "Unknown hue - Shelf Life Prediction unavailable"

    except Exception as e:
        # Log the error and return a meaningful message
        print(f"Error in prediction: {e}")
        return "Error in prediction."

def get_dominant_hue(image_path):
    try:
        # Convert image to HSV and compute dominant hue
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        dominant_hue = np.argmax(hue_hist)
        print(f"Dominant hue extracted: {dominant_hue}")  # Debugging message
        return dominant_hue
    except Exception as e:
        print(f"Error in hue extraction: {e}")  # Debugging message
        return 0  # Default value if error occurs

# --- Real-Time Firebase Data Route ---
@app.route('/firebase_data')
def firebase_data():
    ref = db.reference('sensor_readings')
    data = ref.order_by_key().limit_to_last(1).get()

    if data:
        latest_data = next(iter(data.values()))  # Get the most recent data entry
        methane = latest_data['methane']
        temp = latest_data['temperature']
        hum = latest_data['humidity']

        # Get the prediction based on the latest data
        prediction = predict_from_latest_data()

        return jsonify({
            'prediction': prediction,
            'methane': methane,
            'temperature': temp,
            'humidity': hum
        })

    return jsonify({'error': 'No data available for prediction.'})

if __name__ == '__main__':
    app.run(debug=True)
