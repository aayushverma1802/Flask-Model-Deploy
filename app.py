from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:  # Make sure 'model.pkl' is in the same directory
    model = pickle.load(model_file)

# Initialize the Flask app
app = Flask(__name__)

# Route to render the HTML form
@app.route('/')
def home():
    try:
        # Render the HTML template from the 'templates' folder
        return render_template('index.html')
    except Exception as e:
        return f"Error loading HTML template: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the data from the JSON request
        data = request.get_json()
        features = data.get('features', [])

        # Convert the features to the correct numpy array format
        input_features = np.array([features])

        # Make the prediction using the model
        prediction = model.predict(input_features)

        # Convert the prediction to a string ("true" or "false")
        result = 'Yes' if prediction[0] == 1 else 'NO'

        # Return the prediction result as a JSON response
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
