import flask
from flask import render_template, request, jsonify
from flask import Flask
import flask
import os
import traceback
from fake_news_detection import FakeNewsDetector


# App definition
app = Flask(__name__)

fake_news_detector = FakeNewsDetector(model_path='model/finetuned_BERT_epoch_2.pt')

#check krne ke liye
# a = 'Says his budget provides the highest state funding level in history for education.'
# b = "LeMieux didn't compare Rubio and Obama on an issue such as those listed at the start of the ad -- he said they both used a familiar campaign tactic, throwing bombs about something they didn't have to vote on themselves. The ad provides no explanation for how he compared the two politicians and neglects to note that LeMieux supported Rubio's campaign once Crist left the GOP."
#        # get the predicted stance and the source
# prediction, sconfidence = fake_news_detector.verifyClaim(a,b)
# print("prediction: ", prediction)
# print("confidence: ", confidence) 

@app.route('/predict', methods=['POST'])
def predict():
   try:
       claim = request.json["news"]
       reference = None #@Rushi ka function se source aega
       print(claim)
       # get the predicted stance and the source
       prediction, confidence= fake_news_detector.verifyClaim(claim=claim,reference=reference)
       print("prediction: ", prediction)
       print("confidence: ", confidence)      
       
       return jsonify({
           "prediction":str(prediction),
           "source":str(reference),
       })     
       
   except:
       print("error")
       return jsonify({
           "trace": traceback.format_exc()
           })



if __name__ == "__main__":
   app.run('0.0.0.0', os.environ.get('PORT', 5000), debug=True)