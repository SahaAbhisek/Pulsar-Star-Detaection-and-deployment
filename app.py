import numpy as np
import pickle
from flask import Flask, request, url_for, render_template


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    print(output)
    if output>0.0:
        predicted_label = 'Pulsar Star'
    else:
        predicted_label = 'Not a Pulsar Star'

    return render_template("index.html",
    prediction_text="The prediction is {} and It is a {}".format(output, predicted_label))



if __name__=="__main__":
    app.run(debug=True, port=8001)