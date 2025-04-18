from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# --- Load Model, Scaler, and Label Encoder ---
print("📥 Loading model and preprocessing tools...")
model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')


# --- Home Route ---
@app.route('/')
def home():
    return render_template('homepage.html')


# --- Custom Prediction Page Route ---
@app.route('/custom', methods=['GET', 'POST'])
def custom_predict():
    if request.method == 'POST':
        data = request.get_json()
        methane = np.log1p(data['methane'])  # Apply log transformation
        temperature = data['temperature']
        humidity = data['humidity']

        # Preprocess
        scaled = scaler.transform([[temperature, humidity]])
        input_array = np.array([[methane] + list(scaled[0])])
        prediction = model.predict(input_array)
        predicted_class = le.inverse_transform(prediction)[0]

        return jsonify({'prediction': predicted_class})
    
    # Render the input form page if GET
    return render_template('custom.html')




if __name__ == '__main__':
    app.run(debug=True)
