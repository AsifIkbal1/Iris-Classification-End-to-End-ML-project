from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import pickle
import numpy as np

def home(request):
    return render(request, 'predictor/home.html')

def dashboard(request):
    return render(request, 'predictor/dashboard.html')

def iris_predict(request):
    result = None
    if request.method == 'POST':
        sl = float(request.POST['sepal_length'])
        sw = float(request.POST['sepal_width'])
        pl = float(request.POST['petal_length'])
        pw = float(request.POST['petal_width'])

        scaler = pickle.load(open('/Users/apple/Downloads/Project/Iris-Classification-End-to-End-ML-project/iris_project/predictor/model/scaler.pkl', 'rb'))
        model = pickle.load(open('/Users/apple/Downloads/Project/Iris-Classification-End-to-End-ML-project/iris_project/predictor/model/svc_model.pkl', 'rb'))

        features = np.array([[sl, sw, pl, pw]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        species_map = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
        result = species_map[prediction]

    return render(request, 'predictor/iris_predict.html', {'result': result})
