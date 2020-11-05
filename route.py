from flask import Flask
from flask import json
from flask_cors import CORS, cross_origin
from flask import request
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def make_summary(conf, logs):
    thisdict = [
      {
        "log_line": 'acquire lock=233570404, flags=0x1, tag="View Lock", name=com.android.systemui, ws=null, uid=10037, pid=2227',
        "template": 'acquire lock=<*>, flags=<*>, tag="View Lock", name=<*>, ws=null, uid=<*>, pid=<*>'
      },
      {
        "log_line": 'visible is system.call.count gt 0',
        "template": 'visible is <*> gt <*>'
      },
    ]
    return thisdict


@app.route("/")
@cross_origin()
def hello():
    data = make_summary()
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    print(response.response)
    return response

@app.route("/submit", methods = ['POST'])
@cross_origin()
def parseLog():
    req = request.get_json()
    data = make_summary(req['conf'], req['logs'])
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response;

if __name__ == '__main__':
    app.run(debug=True)
