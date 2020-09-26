from flask import render_template, request, jsonify
from flask import Flask
import flask
import os
import numpy as np
import traceback
import pandas as pd

# App definition
app = Flask(__name__)
fake_detection = FakeDetection()


def get_predictions(text):
    return stance, article


@app.route('/predict', methods=['POST'])
def predict():
   try:
       json_ = request.json
       prediction, source = fake_detection.get_predictions(json_['news'])      
       return jsonify({
           "prediction":str(prediction),
           "source":str(source),
       })     
   except:
       return jsonify({
           "trace": traceback.format_exc()
           })



if __name__ == "__main__":
   app.run('0.0.0.0', os.environ.get('PORT', 5000), debug=True)
