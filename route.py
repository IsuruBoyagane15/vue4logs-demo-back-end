from flask import Flask
import pandas as pd
from flask import json
import ast
from flask_cors import CORS, cross_origin
from flask import request
from Vue4logsParser import *
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def list_logs(logs):
    out = []
    buff = []
    for c in logs:
        if c == '\n':
            out.append(''.join(buff))
            buff = []
        else:
            buff.append(c)
    else:
        if buff:
            out.append(''.join(buff))

    return out


def make_summary(conf, logs):
    conf = ast.literal_eval(conf)
    # print('log_format: ', conf['threshold'])
    logs = list_logs(logs)
    # print(logs[0])

    parser = Vue4Logs(conf, logs)

    pa = parser.parse()
    # print("pa: ",list(pa.T.to_dict().values()))
    thisdict = list(pa.T.to_dict().values())
    
    return thisdict

def save_summary(res, fileName):
    # print('hit',len(list(res.values())))
    df_final = pd.DataFrame()
    for i in list(res.values()):
        print('==============')
        df = pd.DataFrame(i)
        print(df)
        df_final = df_final.append(df[['headers','Content','EventTemplate', 'Log_line']],ignore_index=True)
        print('==============')
    print("--------",df_final,"-------------")
    df_final.to_csv('results/'+fileName+'.csv')
    return "Saves successfully"

@app.route("/", methods=['GET'])
@cross_origin()
def hello():
    response = app.response_class(
        response='Hi',
        status=200,
        mimetype='application/json'
    )
    print("response.response")
    return response


@app.route("/submit", methods=['POST'])
@cross_origin()
def parseLog():
    req = request.get_json()
    
    data = make_summary(req['conf'], req['logs'])
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/save", methods=['POST'])
@cross_origin()
def saveLog():
    req = request.get_json()
    print("request:",req)
    data = save_summary(req['logs'],req['fileName'])
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
