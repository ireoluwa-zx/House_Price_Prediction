from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained pipeline
# Ensure 'model' folder exists and contains the pkl file
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get inputs from form
            features = [
                int(request.form['OverallQual']),
                float(request.form['GrLivArea']),
                float(request.form['TotalBsmtSF']),
                int(request.form['GarageCars']),
                int(request.form['FullBath']),
                int(request.form['YearBuilt'])
            ]
            
            # Convert to numpy array (2D)
            final_features = np.array([features])
            
            # Predict (Pipeline handles scaling automatically)
            prediction = model.predict(final_features)
            output = round(prediction[0], 2)

            return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,.2f}')
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)