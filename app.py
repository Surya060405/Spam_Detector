from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

def remove_punctuation_and_stopwords(text):
    # Add the implementation here
    return text
# Load the trained model
model = joblib.load('Newmodel.pkl')

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Retrieve the input text from the form
            input_text = request.form['message']

            # Check if input_text is empty
            if not input_text.strip():
                return jsonify({'error': 'Please enter a valid message.'})

            # Make a prediction
            prediction = model.predict([input_text])[0]

            # Interpret the result
            result = 'Spam' if prediction == 1 else 'Ham'

            return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

