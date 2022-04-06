import tensorflow as tf
from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
from tensorflow.keras.preprocessing import image   # for preprocessing the images
from tensorflow.keras.utils import np_utils
from werkzeug.utils import secure_filename
import os
from app import app
import urllib.request
from tensorflow.keras.applications.inception_v3 import preprocess_input
    
@app.route('/')

model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)

UPLOAD_FOLDER = 'static/uploads/'

# Initalise the Flask app
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/search_for_obj", methods=["POST"])
def search_for_obj():
    file_names = pd.DataFrame(columns=["names"])
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cap = cv2.VideoCapture(url_for('static', filename='uploads/' + filename))   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                filename ="frame%d.jpg" % count;count+=1
                file_names.append({"names": filename}, ignore_index=True)
                cv2.imwrite(filename, frame)
        cap.release()
        print ("Done!")

        frames = []
        for index, row in file_names.iterrows():
            frames.append(plt.imread(url_for('static', filename='frames/' + row["names"])))
        frames = np.array(frames)

        image = []
        for i in range(0,frames.shape[0]):
            a = resize(frames[i], preserve_range=True, output_shape=(299,299,3)).astype(int)
            image.append(a)
        frames = np.array(image)
        frames = preprocess_input(frames)

        predictions = model.predict(frames)

        object = request.form.get("object")
        classes = pd.read_json("classes.json")

        new_index = list(classes["classes"])
        new_col = list(classes.index)
        new_classes = pd.DataFrame({"class": new_col}, index=new_index)

        clas = ""
        for s in new_index:
            if object in s:
                clas = s
                break

        clas = new_classes.loc[clas]

        frame_nums = []
        for i in range(frames.shape[0]):
            highest = max(predictions[i])
            if clas == list(predictions[i]).index(highest):
                frame_nums.append(i)

        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'frame'+str(frame_num) +'.jpg')
        return render_template("display.html", num = len(frame_nums), user_images = frame_nums)



if __name__ == '__main__':
    app.run(debug=True)