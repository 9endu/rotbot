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
model1=joblib.load('CatBoost_model_20250417_161111.pkl')

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

# Add the route to serve the shelf_life_live.html
@app.route('/shelf-life-data')
def shelf_life_data():
    ref = db.reference('sensor_readings')
    data = ref.order_by_key().limit_to_last(1).get()

    if data:
        latest_data = next(iter(data.values()))  # Get the most recent data entry
        methane = latest_data['methane']
        temp = latest_data['temperature']
        hum = latest_data['humidity']

        prediction = predict_from_latest_data_for_shelflife()

        return render_template('shelf_life_data.html', 
                               prediction=prediction, 
                               methane=methane, 
                               temperature=temp, 
                               humidity=hum)
    
    return render_template('shelf_life_data.html', error="No data available for prediction.")


# Define a mapping from numeric outputs to the human-readable classes
label_mapping = {
    1: "Day 1 ",
    2: "Day 2",
    3: "Day 3",
    4: "Day 4",
    5: "Day 5"
}

def predict_from_latest_data_for_shelflife():
    ref = db.reference('sensor_readings')
    data = ref.order_by_key().limit_to_last(1).get()

    if not data:
        print("⚠️ No data found in Firebase.")
        return None

    for key in data:
        entry = data[key]
        methane = entry['methane']
        temp = entry['temperature']
        hum = entry['humidity']

        print(f"📊 Latest Data - Methane: {methane}, Temp: {temp}, Humidity: {hum}")  # Log the data
        
        # Preprocess
        scaled = scaler.transform([[temp, hum]])
        input_array = np.array([[np.log1p(methane)] + list(scaled[0])])  # Apply log1p for methane
        prediction = model1.predict(input_array)

        # Ensure prediction is a scalar value (convert from ndarray if needed)
        predicted_class = prediction[0] if isinstance(prediction, np.ndarray) else prediction

        # Map numeric prediction to human-readable label
        predicted_label = label_mapping.get(int(predicted_class), "Unknown")  # Convert to int to avoid issues with numpy types
        
        return predicted_label

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join('uploads', file.filename)
            file.save(img_path)

            # Process Image for Banana Detection
            image = cv2.imread(img_path)
            model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            model.eval()

            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image)
            with torch.no_grad():
                prediction = model([image_tensor])[0]
                # 🐛 DEBUG: Print detected classes and their scores
                print("Detected labels:", prediction['labels'].tolist())
                print("Scores:", prediction['scores'].tolist())

            # Find Banana Class ID (COCO: banana = 52)
            for i, label in enumerate(prediction['labels']):
                if label.item() == 52 and prediction['scores'][i].item() > 0.4:
                    mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
                    segmented = cv2.bitwise_and(image, image, mask=mask)

                    # Convert to HSV
                    hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

                    # Define Hue ranges for each color
                    color_ranges = {
                        "Green": (np.array([30, 40, 40]), np.array([80, 255, 255])),
                        "Yellow": (np.array([20, 40, 40]), np.array([30, 255, 255])),
                        "Brown": (np.array([10, 40, 40]), np.array([20, 255, 255]))
                    }

                    # Define Shelf Life for each color
                    shelf_life = {
                        "Green": (7, 10),
                        "Yellow": (3, 5),
                        "Brown": (1, 2)
                    }

                    # ---- Compute Color Percentages ----
                    total_pixels = hsv.size
                    percentages = {}

                    for color, (lower, upper) in color_ranges.items():
                        mask = cv2.inRange(hsv, lower, upper)
                        percentages[color] = (np.count_nonzero(mask) / total_pixels) * 100

                    # Sort colors by percentage
                    sorted_colors = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

                    # If no banana detected
                    if sorted_colors[0][1] == 0:
                        shelf_life_prediction = "No banana detected!"
                    else:
                        top_color, top_value = sorted_colors[0]
                        second_color, second_value = sorted_colors[1]

                        # Determine the shelf life prediction based on top two colors
                        if abs(top_value - second_value) <= 5:
                            # Average the shelf life values if the top two colors are close in percentage
                            avg_min = (shelf_life[top_color][0] + shelf_life[second_color][0]) // 2
                            avg_max = (shelf_life[top_color][1] + shelf_life[second_color][1]) // 2
                            shelf_life_prediction = f"Shelf Life: {avg_min}-{avg_max} days ({top_color} & {second_color})"
                        else:
                            # Use the top color shelf life if the difference is large
                            min_days, max_days = shelf_life[top_color]
                            shelf_life_prediction = f"Shelf Life: {min_days}-{max_days} days ({top_color})"

                    return render_template('image_upload.html', prediction=shelf_life_prediction, dominant_color=top_color)

            return render_template('image_upload.html', error="No banana detected.")

    return render_template('image_upload.html')



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
