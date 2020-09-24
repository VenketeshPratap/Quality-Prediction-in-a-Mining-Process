import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('Multyyllb.save')
trans=load('Transformssllb')
modela= load('Multyylla.save')
transa=load('Transformsslla')

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    test=trans.transform(x_test)
    test=test[:,0:]
    print(test)
    prediction = model.predict(test) 
    print(prediction)
    output=prediction[0] 
    
    return render_template('index.html', prediction_text='The silica(impurity) % without Iron ore concentrate ={}'.format(output))


@app.route('/index2')
def index2():
    return render_template('index2.html')
    


@app.route('/y_predictt',methods=['POST'])
def y_predictt():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    test=transa.transform(x_test)
    test=test[:,:]
    print(test)
    prediction= modela.predict(test) 
    print(prediction)
    f=str(prediction)
    f=f.split(' ')
    m=f[0]
    nm=m.strip("[[")
    k=f[2]
    pf=k.strip("]]")
 
    
    
    
    return render_template('index2.html', prediction_text='The silica(impurity)='+pf+' with Iron ore concentrate='+nm+'%')
                           
@app.route('/about3')
def about3():
    return render_template('about3.html')
    
'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
