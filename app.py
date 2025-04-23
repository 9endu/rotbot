import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import pyrebase
from torchvision import models, transforms

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory,flash

# Firebase (if used elsewhere in your app)
import firebase_admin
from firebase_admin import credentials, db

# Optional utilities (if you still use them in your app)
import datetime
import re
import serial
import threading
import joblib


# Initialize Flask app
app = Flask(__name__)
app.secret_key ='a3443ca5af1d56ae7bd937cc8d2c462d'

ADMIN_EMAIL = 'admin@example.com'
ADMIN_PASSWORD = 'adminpass'


# Firebase Initialization
cred = credentials.Certificate("firebase/rotbot-b300b-firebase-adminsdk-fbsvc-3b9c6a4580.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rotbot-b300b-default-rtdb.firebaseio.com/'
})

firebaseConfig = {
    "apiKey": "AIzaSyAOk65Mg_P1ISLbGRB6I-3HDL-U-hCFO-c",
    "authDomain": "rotbot-b300b.firebaseapp.com",
    "databaseURL": "https://rotbot-b300b-default-rtdb.firebaseio.com",
    "projectId": "rotbot-b300b",
    "storageBucket": "rotbot-b300b.appspot.com",
    "messagingSenderId": "1022382810982",
    "appId": "1:1022382810982:web:cb0f1a631f857de4a18aa7",
    "measurementId": "G-B6S4DC2L03"
}

import pyrebase

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()


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
@app.route('/')
def home():
    # If user is logged in, redirect to homepage, else show login page
    if 'user' in session:
        return redirect('/homepage')
    else:
        return redirect('/login')  # Redirect to login page if not logged in

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = {
            "company_name": request.form['company_name'],
            "industry": request.form['industry'],
            "location": request.form['location'],
            "name": request.form['name'],
            "email": request.form['email'],
            "phone": request.form['phone'],
            "password": request.form['password']
        }

        email_key = data['email'].replace('.', ',')
        ref = db.reference('companies')
        ref.child(email_key).set(data)

        return redirect('/login')  # Redirect to login page after signup
    return render_template('signup.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        # Access the 'companies' node in the Firebase Realtime Database
        companies_ref = db.reference('companies')  
        companies = companies_ref.get()  # Get all companies data

        # Iterate through the companies to check if the email exists
        for company_key, company_data in companies.items():
            if company_data.get('email') == email:
                return render_template('reset_password.html', email=email)

        return "Email not found. Try again."

    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['POST'])
def reset_password():
    email = request.form['email']
    new_password = request.form['new_password']

    # Access the 'companies' node in the Firebase Realtime Database
    companies_ref = db.reference('companies')
    companies = companies_ref.get()  # Get all companies data

    # Iterate through the companies to find the email and update the password
    for company_key, company_data in companies.items():
        if company_data.get('email') == email:
            # Update the password for the company
            company_ref = companies_ref.child(company_key)  # Reference to the specific company
            company_ref.update({
                "password": new_password
            })

            # Redirect to login.html with a success message
            return redirect(url_for('login', success_message="Password updated successfully"))

    return "Something went wrong."

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')

    email_key = session['user'].replace('.', ',')
    user_data = db.reference(f'companies/{email_key}').get()

    return render_template('profile.html', user=user_data)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user' not in session:
        return redirect('/login')

    email_key = session['user'].replace('.', ',')
    ref = db.reference(f'companies/{email_key}')

    updated_data = {
        'company_name': request.form['company_name'],
        'industry': request.form['industry'],
        'location': request.form['location'],
        'name': request.form['name'],
        'phone': request.form['phone'],
        'email': session['user']  # don't update email
    }

    ref.update(updated_data)
    flash('Profile updated successfully!', 'success')
    return redirect('/profile')

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' not in session:
        return redirect('/login')

    email_key = session['user'].replace('.', ',')
    ref = db.reference(f'companies/{email_key}')
    user_data = ref.get()

    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']

    if user_data['password'] != current_password:
        flash('Current password is incorrect.', 'error')
    elif new_password != confirm_password:
        flash('New passwords do not match.', 'error')
    elif len(new_password) < 8:
        flash('Password must be at least 8 characters.', 'error')
    else:
        ref.update({'password': new_password})
        flash('Password changed successfully!', 'success')

    return redirect('/profile')

@app.route('/delete_account')
def delete_account():
    if 'user' not in session:
        return redirect('/login')

    email_key = session['user'].replace('.', ',')
    db.reference(f'companies/{email_key}').delete()

    session.clear()
    flash('Account deleted successfully.', 'success')
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        email_key = email.replace('.', ',')

        # Check if admin
        if email == 'admin@example.com' and password == 'adminpass':
            session['admin'] = True
            return redirect('/admin')

        # Check regular user
        ref = db.reference(f'companies/{email_key}')
        user_data = ref.get()

        if user_data:
            if user_data['password'] == password:
                session['user'] = user_data['email']
                session['name'] = user_data['name']
                return redirect('/homepage')
            else:
                error = "Incorrect password"
        else:
            error = "User does not exist"

    return render_template('login.html', error=error)


@app.route('/homepage')
def homepage():
    if 'user' in session:
        # Pass the user's name from the session to the template
        return render_template('homepage.html', user=session['user'], name=session['name'])
    else:
        return redirect('/login')  # Redirect to login if not logged in
    
@app.route('/logout')
def logout():
    # Clear the session to log the user out
    session.pop('user', None)
    session.pop('name', None)  # Remove the name from session as well
    return redirect('/login')  # Redirect to the login page

@app.route('/admin')
def admin():
    # Ensure the admin is logged in
    if not session.get('admin'):
        return redirect('/login')

    try:
        # Get Firebase references
        user_ref = db.reference('companies')
        custom_data_ref = db.reference('custom_data')
        firebase_data_ref = db.reference('firebase_data')

        # Fetch data or return empty dicts if nothing exists
        users = user_ref.get() or {}
        custom_data = custom_data_ref.get() or {}
        firebase_data = firebase_data_ref.get() or {}

        # Render admin dashboard with all data
        return render_template(
            'admin.html',
            users=users,
            custom_data=custom_data,
            firebase_data=firebase_data
        )

    except Exception as e:
        print(f"Error fetching admin data: {e}")
        return "An error occurred while loading admin data.", 500

# --- Firebase Prediction Route ---
from flask import session, render_template
import datetime
import firebase_admin
from firebase_admin import credentials, db

# --- Firebase Prediction Route ---
@app.route('/firebase', methods=['GET', 'POST'])
def firebase_predict():
    prediction = predict_from_latest_data()

    if prediction:
        # Fetch the latest sensor data from Firebase
        ref = db.reference('sensor_readings')
        data = ref.order_by_key().limit_to_last(1).get()

        if data:
            latest_data = next(iter(data.values()))
            methane = latest_data['methane']
            temp = latest_data['temperature']
            hum = latest_data['humidity']

            # Get the logged-in user's email
            user_email = session.get('user')
            if not user_email:
                return render_template('firebase.html', error="User not logged in")

            email_key = user_email.replace('.', ',')  # Firebase key-friendly format

            # Reference to user's firebase_data history
            data_ref = db.reference(f'companies/{email_key}/history/firebase_data')

            # Check the last saved entry
            existing_entries = data_ref.order_by_key().limit_to_last(1).get()
            if existing_entries:
                last_entry = next(iter(existing_entries.values()))
                if (last_entry['methane'] == methane and
                    last_entry['temperature'] == temp and
                    last_entry['humidity'] == hum):
                    # No change in data, do not log again
                    return render_template('firebase.html', prediction=prediction, methane=methane, temperature=temp, humidity=hum)

            # New data detected — log it
            prediction_data = {
                "methane": methane,
                "temperature": temp,
                "humidity": hum,
                "output": prediction,
                "timestamp": datetime.datetime.now().isoformat()
            }

            data_ref.push(prediction_data)  # Push new prediction entry

            return render_template('firebase.html', prediction=prediction, methane=methane, temperature=temp, humidity=hum)

    return render_template('firebase.html', error="No data available for prediction.")

@app.route('/track_history')
def track_history():
    # Ensure the user is logged in
    if not session.get('user'):
        return redirect('/login')

    user_email = session['user'].replace('.', ',')  # Match Firebase format

    try:
        # Get Firebase references
        user_ref = db.reference(f'companies/{user_email}')  # Get user-specific data
        user_data = user_ref.get()

        if user_data:
            custom_data = user_data.get('history', {}).get('custom_data', {})
            firebase_data = user_data.get('history', {}).get('firebase_data', {})
        else:
            custom_data = {}
            firebase_data = {}

        # Render track history page with user-specific data
        return render_template(
            'track_history.html',
            custom_data=custom_data,
            firebase_data=firebase_data
        )

    except Exception as e:
        print(f"Error fetching history data: {e}")
        return "An error occurred while loading history data.", 500

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

        # Get user email from session
        user_email = session.get('user')
        email_key = user_email.replace('.', ',')  # Firebase does not allow '.' in keys
        
        # Check if the 'history' node exists, if not, create it
        ref = db.reference(f'companies/{email_key}/history')
        if ref.get() is None:  # If 'history' does not exist, create it
            ref.set({
                'custom_data': []  # Create an empty list for custom data
            })

        # Prepare data for saving to Firebase
        prediction_data = {
            "methane": data['methane'],  # Original methane value
            "temperature": temperature,
            "humidity": humidity,
            "output": predicted_class,
            "timestamp": datetime.datetime.now().isoformat()  # Current timestamp
        }

        # Save the prediction data to Firebase under the user's history
        ref = db.reference(f'companies/{email_key}/history/custom_data')
        ref.push(prediction_data)

        return jsonify({'prediction': predicted_class})

    return render_template('custom.html')

@app.route('/shelf-life-data')
def shelf_life_data():
    ref = db.reference('sensor_readings')
    data = ref.order_by_key().limit_to_last(1).get()

    if data:
        latest_data = next(iter(data.values()))
        methane = latest_data['methane']
        temp = latest_data['temperature']
        hum = latest_data['humidity']

        prediction = predict_from_latest_data_for_shelflife()

        # Get user email from session
        user_email = session.get('user')
        if user_email:
            email_key = user_email.replace('.', ',')
            history_ref = db.reference(f'companies/{email_key}/history')

            if history_ref.get() is None:
                history_ref.set({
                    'shelf_life_data': []  # Create empty list for shelf life data
                })

            # Prepare data for saving
            prediction_data = {
                "methane": methane,
                "temperature": temp,
                "humidity": hum,
                "output": prediction,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Save to Firebase under shelf_life_data
            data_ref = db.reference(f'companies/{email_key}/history/shelf_life_data')
            data_ref.push(prediction_data)

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

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory('uploads', filename)

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('image_upload.html', error="No file selected.")

        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Read and validate image
        image = cv2.imread(img_path)
        if image is None:
            return render_template('image_upload.html', error="Invalid image. Please try again.")

        # Start timer
        start_time = time.time()

        # Load model
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        # Transform and predict
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            image_tensor = transform(image)
        except Exception as e:
            return render_template('image_upload.html', error="Error processing image.")
        with torch.no_grad():
            prediction = model([image_tensor])[0]
        print("Detected labels:", prediction['labels'].tolist())
        print("Scores:", prediction['scores'].tolist())

        for i, label in enumerate(prediction['labels']):
            if label.item() == 52 and prediction['scores'][i].item() > 0.4:
                mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
                segmented = cv2.bitwise_and(image, image, mask=mask)
                hsv = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)

                color_ranges = {
                    "Green": (np.array([30, 40, 40]), np.array([80, 255, 255])),
                    "Yellow": (np.array([20, 40, 40]), np.array([30, 255, 255])),
                    "Brown": (np.array([10, 40, 40]), np.array([20, 255, 255]))
                }

                shelf_life = {
                    "Green": (7, 10),
                    "Yellow": (3, 5),
                    "Brown": (1, 2)
                }

                total_pixels = hsv.size
                percentages = {}

                for color, (lower, upper) in color_ranges.items():
                    mask = cv2.inRange(hsv, lower, upper)
                    percentages[color] = (np.count_nonzero(mask) / total_pixels) * 100

                sorted_colors = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

                if sorted_colors[0][1] == 0:
                    shelf_life_prediction = "No banana detected!"
                    dominant_color = "N/A"
                else:
                    top_color, top_value = sorted_colors[0]
                    second_color, second_value = sorted_colors[1]

                    if abs(top_value - second_value) <= 5:
                        avg_min = (shelf_life[top_color][0] + shelf_life[second_color][0]) // 2
                        avg_max = (shelf_life[top_color][1] + shelf_life[second_color][1]) // 2
                        shelf_life_prediction = f"Shelf Life: {avg_min}-{avg_max} days ({top_color} & {second_color})"
                    else:
                        min_days, max_days = shelf_life[top_color]
                        shelf_life_prediction = f"Shelf Life: {min_days}-{max_days} days ({top_color})"

                    dominant_color = top_color

                # End timer
                prediction_time = round(time.time() - start_time, 2)

                return render_template(
                    'image_upload.html',
                    prediction=shelf_life_prediction,
                    dominant_color=dominant_color,
                    prediction_time=prediction_time
                )

        return render_template('image_upload.html', error="No banana detected. Try uploading or capturing again.")

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
