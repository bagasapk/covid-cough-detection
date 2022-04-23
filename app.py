from flask import Flask
from datetime import datetime
import re
from flask import render_template


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('contact.html')

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
