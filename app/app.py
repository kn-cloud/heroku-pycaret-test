from flask import Flask,render_template,request,jsonify,url_for,render_template
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('final_model')
cols = ['K_Reuss', 'K_VRH', 'K_Voigt', 'elastic_anisotropy']

@app.route("/")
@app.route("/index")
def index():
    name = request.args.get("name")
    return render_template("index.html", name=name)

@app.route("/index",methods=["post"])
def post():
    features = [x for x in request.form.values()]
    print(features)
    final = np.array(features)
    print(final)
    data_unseen = pd.DataFrame([final], columns = cols)
    print(data_unseen)
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.Label[0]

    return render_template("index.html", \
        K_Reuss=features[0], K_VRH=features[1], K_Voigt=features[2], elastic_anisotropy=features[3], \
        pred = 'Expected K_Voigt_Reuss_Hill will be {}'.format(prediction))

@app.route('/predict_api', methods=["post"])
def predict_api():
    data = request.get_json(force=True)
    print(data)

    return jsonify(42.1)

if __name__ == "__main__":
    app.run(debug=True)
