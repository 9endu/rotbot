import serial
import time
import threading
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import datetime
import re
from flask import Flask, render_template, request, jsonify

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
