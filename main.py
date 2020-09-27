import flask
from flask import render_template, request, jsonify
from flask import Flask
import flask
# from flask_ngrok import run_with_ngrok

import os
import traceback
from NEWS.fake_news_detection import FakeNewsDetector
from NEWS.solver import Solver

# App definition
app = Flask(__name__)
# run_with_ngrok(app)
model_path = 'model/finetuned_BERT_epoch_5.pt'
fake_news_detector = FakeNewsDetector(model_path=model_path)


@app.route('/predict', methods=['POST'])
def predict():
   try:
       claim = request.form['news']
    #    claim = data['news']
       print(claim)
       
       # get the predicted stance and the source
       try:
        (prediction, source) = Solver(claim, fake_news_detector)
       except Exception as e:
           print("None returned: ", e)
           prediction = "unrelated"
           source = "The question is unrelated to any new article we currently have"      
       
       return jsonify({
           "prediction": str(prediction),
           "source": str(source),
       })     
       
   except Exception as e:
       print("error: ", e)
       return jsonify({
           "trace": traceback.format_exc()
           })



if __name__ == "__main__":
   app.run('0.0.0.0', os.environ.get('PORT', 5000), debug=True)
    # app.run()