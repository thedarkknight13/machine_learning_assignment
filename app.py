import tensorflow as tf
from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
from tensorflow.keras.preprocessing import image   # for preprocessing the images
from keras.utils import np_utils
from werkzeug.utils import secure_filename
import os, shutil
import urllib.request
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
    
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

data = pd.read_json("imagenet_class_index.json")

nums = list(data.columns)
classes = list(data.loc[1])

num_to_class = pd.DataFrame({"classes":classes}, index=nums)
class_to_num = pd.DataFrame({"nums":nums}, index=classes)

@app.route('/')
def home():
    return render_template("home.html", error=None, message=None)

@app.errorhandler(413)
def file_to_large(error):
    return render_template("home.html", error=True, message="The maximum size of the videos should be 2MB")

@app.route("/search_for_obj", methods=["POST"])
def search_for_obj():
    file_names = pd.DataFrame(columns=["names"])
    if 'file' not in request.files:
        return render_template("home.html", error=True, message="You are supposed to upload a video file!")
    file = request.files['file']
    if file.filename == '':
        return render_template("home.html", error=True, message="You are supposed to upload a video file!")
    else:
        obj = request.form.get("object")
        # Checking if the obj value is empty
        classe = None
        if obj == "" or obj is None:
            return render_template("home.html", error=True, message="Object for query should not be blank")
        else:
            for c in classes:
                if obj.lower().replace(" ", "_") in c.split(","):
                    classe = c
                    break
            if not classe:
                return render_template("home.html", error=True, message="Our model was trained with imagenet classes. We are sorry we cannot identify the object you want. You can try another one...")

        # First delete the frames currently in the frames folder
        folder = 'static/frames'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        # Then delete the videos currently in the uploads folder
        folder = 'static/uploads'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Separating the video into frames
        cap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename) )  # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        count = 0
        success, image = cap.read()
        for i in range(10):
            cv2.imwrite(os.path.join("static/frames/", "frame%d.jpg" % count), image)
            file_names = file_names.append({"names":"frame%d.jpg" % count}, ignore_index=True)
            success,image = cap.read()
            count += 1
        
        # Reading the frames from the frames folder
        frames = []
        for index, row in file_names.iterrows():
            frames.append(plt.imread(os.path.join('static/frames/', row["names"])))
        frames = np.array(frames)

        # Preprocssing the frames for the inceptionV3 model
        image = []
        for i in range(0,frames.shape[0]):
            a = resize(frames[i], preserve_range=True, output_shape=(299,299,3)).astype(int)
            image.append(a)
        frames = np.array(image)
        frames = preprocess_input(frames)

        # Getting the predictions for the frames
        predictions = model.predict(frames)

        # Finding the frames that include the object
        
        num = class_to_num.loc[classe][0]

        images = []
        for i in range(predictions.shape[0]):
            highest = max(predictions[i])
            if num == list(predictions[i]).index(highest):
                images.append(os.path.join("static/frames/", 'frame'+str(i) +'.jpg'))
        return render_template("display.html", num = len(images), user_images = images)


@app.route("/back_home")
def go_back_home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)
