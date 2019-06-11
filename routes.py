import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
#Simport pickle

from utils import onehotCategorical

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        entered_li = []

        # YOUR CODE FOR PART 2.2
        # get request values
        quarter = int(request.form['Quarter'])
        minute = int(request.form['Minute'])
        second = int(request.form['Seconds'])
        score = int(request.form['Score'])
        fav = int(request.form['Favorite'])
        home = int(request.form['Home'])
        ps = int(request.form['PointSpread'])
        half = float(request.form['Half'])
        down = int(request.form['Down'])
        distance = int(request.form['Distance']) / 10
        field = abs(int(request.form['Field']) - 100) / 100
        
        time_left = (3600 - (quarter * 900) + (minute * 60) + second) / 3600
        if fav == 1:
            posteam_spread = -1 * (ps + half)
        else:
            posteam_spread = (ps + half)

            
        # one-hot encode categorical variables
        down_encode = onehotCategorical(down, 4)

        # engineer 1 observation for prediction
        # YOUR CODE START HERE
        entered_li.extend(down_encode)
        entered_li.extend([distance])
        entered_li.extend([field])
        entered_li.extend([score])
        entered_li.extend([home])
        entered_li.extend([time_left])
        entered_li.extend([posteam_spread])
        
        #pkl_file = open('lr.pkl', 'rb')
        #model = pickle.load(pkl_file)
        model = joblib.load('lr.pkl')
        prediction_o = model.predict_proba(np.array(entered_li).reshape(1, -1))[0][1] * 100
        prediction_d = 100 - prediction_o
        #prediction = model.predict(entered_li.values.reshape(1, -1))
        
        if (time_left == 0) & (score > 0):
            label_o = '100.00'
            label_d = '00.00'
            
        elif (time_left == 0) & (score < 0):
            label_o = '00.00'
            label_d = '100.00'
            
        elif (0 < time_left < (42 / 3600)) & (score > 0):
            label_o = '99.99'
            label_d = '0.01'
            
        elif (0 < time_left < (61 / 3600)) & (score > 0):
            label_o = '99.49'
            label_d = '0.51'
            
        elif (0 < time_left < (42 / 3600)) & (score < -8):
            label_o = '0.01'
            label_d = '99.99'
        
        elif (0 < time_left < (61 / 3600)) & (score < -8):
            label_o = '0.51'
            label_d = '99.49'
            
        else:
            label_o = str(np.squeeze(prediction_o.round(2)))
            label_d = str(np.squeeze(prediction_d.round(2)))
            #label_o = str(prediction_o * 100)[:5]
            #label_d = str(prediction_d * 100)[:5]

        return render_template('index.html', labelo = label_o, labeld = label_d)

if __name__ == '__main__':
    # load ML model
    #model = joblib.load('lr.pkl')
    #pkl_file = open('lr.pkl', 'rb')
    #model = pickle.load(pkl_file)
    # start API
    app.run(host='0.0.0.0', port=8000, debug=True)
    #app.run(debug=True)
