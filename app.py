from flask import Flask
from datetime import datetime
import re
import os
import uuid
import io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import render_template,flash,request,redirect,Response,session
from convert_audio_2_spec import convert_audio_to_mel
from convert_audio_2_spec import preprocess
from convert_audio_2_spec import generategraph
from convert_audio_2_spec import get_melspec
from convert_audio_2_spec import printPredict

UPLOAD_FOLDER = 'static/files'

app = Flask(__name__)
app.secret_key = 'SECRET_KEY'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template('test.html')

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")

@app.route('/',methods = ['POST'])
def save_record():
    if 'file' not in request.files:
        flash('No File Part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No Selected File')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    feature = convert_audio_to_mel(full_file_name)
    feature = preprocess(feature,featureSize = (224,224))
    fig = generategraph(file_name,feature)
    feature = get_melspec(file_name)
    feature = feature.reshape(-1,224,224,3)
    session['name'] = []
    session['name'].append(file_name)
    # printPredict(feature)
    return "Success"