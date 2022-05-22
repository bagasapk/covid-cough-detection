from datetime import datetime
import re
import os
import uuid
import io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import render_template,flash,request,redirect,Response,session,url_for,Flask,send_from_directory
from convert_audio_2_spec import convert_audio_to_mel
from convert_audio_2_spec import preprocess
from convert_audio_2_spec import generategraph
from convert_audio_2_spec import get_melspec
from convert_audio_2_spec import printPredict

UPLOAD_FOLDER = 'static/files'
UPLOAD_PLT = "static/plt"

app = Flask(__name__)
app.secret_key = "b'\xcapZ\x14~`\xb9\x8e\xa3\xa9\xa5>'"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_PLT'] = UPLOAD_PLT

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

@app.route('/',methods = ['GET','POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No File Part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No Selected File')
            return redirect(request.url)
        filename = str(uuid.uuid4()) + ".wav"
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_filename)
        feature = convert_audio_to_mel(full_filename)
        feature = preprocess(feature,featureSize = (224,224))
        generategraph(filename,feature)
        feature = get_melspec(filename)
        feature = feature.reshape(-1,224,224,3)
        image_path = os.path.join(app.config['UPLOAD_PLT'], filename)
        output = printPredict(feature)
        messages = image_path+'.png'
        session['messages'] = messages
        session['output'] = output
        return redirect(url_for('success', messages = messages, output = output))
    return render_template('test.html')

@app.route("/success")
def success():
    # messages = request.args['messages']  # counterpart for url_for()
    messages = session['messages'] 
    output = session['output']
    return render_template('success.html', messages = messages, output = output)

@app.route('/model/<path:path>')
def send_report(path):
    return send_from_directory('static', path)