import tensorflow as tf
import gradio as gr
import requests
import urllib
from PIL import Image
import numpy as np



model = tf.keras.models.load_model("my_model.h5")
model.summary()

labels = ["No Wildfire", "Wildfire"]

#This function will preprocess the data
#img should be np array


def process(img):

# asarray() class is used to convert
# PIL images into NumPy arrays
    img = np.asarray(img)
    #img_size --> 350,350,3
    new_img = np.reshape(img, (-1,350,350,3))
    #Batch Normalization
    new_img = new_img / 255


    return new_img



def classify_img(inp):
    inp = inp[None, ...]
    inp = process(inp)
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(2)}

image = gr.inputs.Image(shape = (350,350))
label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=classify_img, inputs= image, outputs=label, interpretation="default").launch(share = True, debug=True)