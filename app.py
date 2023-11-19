from flask import Flask

app =  Flask("preditor-de-avc")
from routes import *

if __name__ == "__main__":
    app.run("0.0.0.0", 3000, debug=True)