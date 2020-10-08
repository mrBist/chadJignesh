from flask import render_template, request, jsonify
from flask import Flask
# from flask_ngrok import run_with_ngrok

import os
import re
import cv2
import pytesseract
import numpy as np
import traceback
from NEWS.fake_news_detection import FakeNewsDetector
from NEWS.solver import Solver

# App definition
app = Flask(__name__)
# run_with_ngrok(app)
model_path = 'model/finetuned_BERT_epoch_5.pt'
fake_news_detector = FakeNewsDetector(model_path=model_path)


@app.route('/predict-text', methods=['POST'])
def predict_text():
    try:
        claim = request.form['news']
        print(claim)
        get_predicted_stance(claim=claim)     
    except Exception as e:
        print("error: ", e)
        return jsonify({
            "trace": traceback.format_exc()
            })


@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        file = request.files['file']
        img = np.fromfile(file, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = process_image(img)
        claim = image_to_text(img=img)
        get_predicted_stance(claim=claim)
    except Exception as e:
        print("error: ", e)
        return jsonify({
            "trace": traceback.format_exc()
            })

def process_image(img):
    # TODO add path
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    scale_percent = 140 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # reduce noise
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    dst = cv2.fastNlMeansDenoisingColored(rgb_img,None,10,10,7,21)
    b,g,r = cv2.split(dst)
    img = cv2.merge([r,g,b])

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    #thresholding
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # deskew the image
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1] 
    if angle < -45:
	    angle = -(90 + angle)
    else:
	    angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # blur & sharpen the image
    img=cv2.GaussianBlur(rotated,(3,3),0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(rotated, -1, kernel)
    return img


def image_to_text(img):
    # convert image to text
    text = pytesseract.image_to_string(img)
    text = re.sub(' +', ' ', text) 
    return text
    

def get_predicted_stance(claim):
    # get the predicted stance and the source
    try:
        try:
         (prediction, source) = Solver(claim, fake_news_detector)
        except Exception as e:
            print("None returned: ", e)
            prediction = "unrelated"
            source = "The question is unrelated to any news article we currently have"      

        return jsonify({
            "prediction": str(prediction),
            "source": str(source),
        })
    except:
        print("error: ", e)
        return jsonify({
           "trace": traceback.format_exc()
           })


if __name__ == "__main__":
   app.run('0.0.0.0', os.environ.get('PORT', 5000), debug=True)
    # app.run()