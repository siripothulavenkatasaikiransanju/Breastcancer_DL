
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
from keras.models import load_model
import pickle

app = Flask(__name__) # initializing a flask app

# Load the trained model
model = load_model('BreastcancerpredicitonDL.h5')

# Load the scaler
with open('Scaler.sav', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            radius_mean=float(request.form['radius_mean'])
            texture_mean=float(request.form['texture_mean'])
            perimeter_mean=float(request.form['perimeter_mean'])
            area_mean=float(request.form['area_mean'])
            smoothness_mean=float(request.form['smoothness_mean'])
            compactness_mean=float(request.form['compactness_mean'])
            concavity_mean=float(request.form['concavity_mean'])
            concave_points_mean=float(request.form['concave_points_mean'])
            symmetry_mean=float(request.form['symmetry_mean'])
            fractal_dimension_mean=float(request.form['fractal_dimension_mean'])
            radius_se=float(request.form['radius_se'])
            texture_se=float(request.form['texture_se'])
            perimeter_se=float(request.form['perimeter_se'])
            area_se=float(request.form['area_se'])
            smoothness_se=float(request.form['smoothness_se'])
            compactness_se=float(request.form['compactness_se'])
            concavity_se=float(request.form['concavity_se'])
            concave_points_se=float(request.form['concave_points_se'])
            symmetry_se=float(request.form['symmetry_se'])
            fractal_dimension_se=float(request.form['fractal_dimension_se'])
            radius_worst=float(request.form['radius_worst'])
            texture_worst=float(request.form['texture_worst'])
            perimeter_worst=float(request.form['perimeter_worst'])
            area_worst=float(request.form['area_worst'])
            smoothness_worst=float(request.form['smoothness_worst'])
            compactness_worst=float(request.form['compactness_worst'])
            concavity_worst=float(request.form['concavity_worst'])
            concave_points_worst=float(request.form['concave_points_worst'])
            symmetry_worst=float(request.form['symmetry_worst'])
            fractal_dimension_worst=float(request.form['fractal_dimension_worst'])

            
            #filename = 'Breastcancerprediciton.h5'
            #loaded_model = load_model('Breastcancerprediciton.keras') # loading the model file from the storage
            # predictions using the loaded model file
            #scaler = pickle.load(open('Scaler.sav', 'rb'))
            prediction_proba = model.predict(scaler.transform([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]]))
            print('prediction probabilities:', prediction_proba)

            # Convert probabilities to class labels based on threshold (0.5)
            if prediction_proba >= 0.5:
                pred = "Benign"
            else:
                pred = "Malignant"

            # showing the prediction results in a UI
            return render_template('results.html', prediction=pred)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app