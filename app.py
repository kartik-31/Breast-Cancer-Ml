import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl','rb'))
scaler=pickle.load(open("scaler.pkl","rb"))
pca=pickle.load(open("dimension_reduction.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    input_features = np.array(int_features)
    input_features=input_features.reshape(1,-1)
    fin_features=scaler.transform(input_features)
    final_features=pca.transform(fin_features)
    prediction = model.predict(final_features)
    output=prediction
    print(output)
    if output==0:
        output="NO CANCER"
    else:
        output="HAVE CANCER"
        
    return render_template('index.html',prediction_text="Prediction is  {}".format(output))
    
    
if __name__ == '__main__':
    app.run()


#<input type="text" name='bare_nuclei' placeholder='bare_nuclei' required="required">