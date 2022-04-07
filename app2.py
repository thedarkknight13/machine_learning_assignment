from flask import Flask,request, url_for, redirect, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html", error=None, message=None)

if __name__ == '__main__':
    app.run(debug=True)