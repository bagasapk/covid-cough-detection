import librosa
import cv2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import random
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
import os

UPLOAD_PLT = "static/plt"

app = Flask(__name__)
app.config['UPLOAD_PLT'] = UPLOAD_PLT
model = load_model('weights-improvement-13-0.84.hdf5')

def white_noise(y):
    wn = np.random.randn(len(y))
    y_wn = y + random.uniform(0, 0.005)*wn
    return y_wn

def time_shift(y):
    y = np.roll(y, random.randint(-10000,10000))
    return y

def Gain(y):
    y = y + random.uniform(-0.2,0.2)*y
    return y       
                            
def stretch(y, rate=random.uniform(0.8,1.2)):
    y = librosa.effects.time_stretch(y, rate)
    return y

def convert_audio_to_mel(filename):
    y,sr = librosa.load(filename, mono=True, duration=10)
    for i in range(1):
        chance = random.randint(0,100)
        if chance <=20:
            stretch(y)
        if chance <=40:
            time_shift(y)
        if chance <=60:
            Gain(y)
        if chance <=80:
            white_noise(y)
        melspec = librosa.feature.melspectrogram(y, sr=sr,n_fft=2048, hop_length=512,n_mels=128)
        feature = librosa.power_to_db(melspec, ref=np.max)
    return feature

def get_melspec(filename):
    full_file_name = os.path.join(app.config['UPLOAD_PLT'], filename)
    im = plt.imread(full_file_name+'.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im


def preprocess(feature, featureSize):
    scaler = StandardScaler()
    widthTarget, heightTarget = featureSize #224,224
    height, width = feature.shape[0], feature.shape[1] #1193,431
  
    newSize = (224, 224) #107,41
    feature = cv2.resize(feature, newSize) 
    # Normalization
    feature = scaler.fit_transform(feature)
    return feature

def generategraph(filename,file):
    fig = plt.figure(figsize=(2.24,2.24))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # plt.axis('off')
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.set_cmap('hot')
    savefile = filename + '.png'  # file might need to be replaced by a string
    full_file_name = os.path.join(app.config['UPLOAD_PLT'], savefile)
    plt.imshow(file)
    plt.savefig(full_file_name, bbox_inches='tight',pad_inches=0)
    return savefile 

def printPredict(feature):
    y_pred = model.predict(feature)
    for i,label in enumerate(y_pred):
        if label <= 0.5:
            predicted = 'Negative'
        else :
            predicted = 'Positive'
    return 'Predicted:{0} {1} Covid-19'.format(label,predicted)