from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from django.http import JsonResponse
import joblib
import os



def home(request):
    return render(request, 'home.html')

def menu(request):
    return render(request, 'menu.html')

def predict(request):
    return render(request, 'predict.html')

def negative(request):
    return render(request, 'negative.html')

def malignant(request):
    return render(request, 'malignant.html')

def positive(request):
    return render(request, 'positive.html')

def Recomm(request):
    return render(request, 'Recomm.html')

def heart(request):
    return render(request, 'heart.html')

def breast(request):
    return render(request, 'breast.html')

def kidney(request):
    return render(request, 'kidney.html')

def DiabetesTreat(request):
    return render(request, 'DiabetesTreat.html')

def HeartTreat(request):
    return render(request, 'HeartTreat.html')

def BreastTreat(request):
    return render(request, 'BreastTreat.html')

def KidneyTreat(request):
    return render(request, 'KidneyTreat.html')

def NegativeDiabetes(request):
    return render(request, 'NegativeDiabetes.html')

def PositiveDiabetes(request):
    return render(request, 'PositiveDiabetes.html')

def NegativeCKD(request):
    return render(request, 'NegativeCKD.html')

def PositiveCKD(request):
    return render(request, 'PositiveCKD.html')

def NegativeHeart(request):
    return render(request, 'NegativeHeart.html')

def PositiveHeart(request):
    return render(request, 'PositiveHeart.html')

def ResultBenign(request):
    return render(request, 'ResultBenign.html')

def load_model_and_scaler():
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "static", "diabetespredict", "models"
    )
    model_path = os.path.join(base_path, "diabetes_model.pkl")
    scaler_path = os.path.join(base_path, "scaler.pkl")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_breast_model_and_scaler():
    # Define the paths for the model and scaler
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "diabetespredict", "models"
    )
    breast_model_path = os.path.join(base_path, "new_breast_model.pkl")
    breast_scaler_path = os.path.join(base_path, "new_breast_scaler.pkl")

    # Load the model and scaler
    new_breast_model = joblib.load(breast_model_path)
    new_breast_scaler = joblib.load(breast_scaler_path)
    return new_breast_model, new_breast_scaler

def load_heart_model_and_scaler():
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "static", "diabetespredict", "models"
    )
    heart_model_path = os.path.join(base_path, "heart_model.pkl")
    heart_scaler_path = os.path.join(base_path, "heart_scaler.pkl")
    
    # Load model and scaler
    heart_model = joblib.load(heart_model_path)
    heart_scaler = joblib.load(heart_scaler_path)
    return heart_model, heart_scaler

def load_kidney_model_and_scaler():
    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "static", "diabetespredict", "models"
    )
    kidney_model_path = os.path.join(base_path, "kidney_model.pkl")
    kidney_scaler_path = os.path.join(base_path, "kidney_scaler.pkl")
    
    # Load model and scaler
    kidney_model = joblib.load(kidney_model_path)
    kidney_scaler = joblib.load(kidney_scaler_path)
    return kidney_model, kidney_scaler



#!Read the data from the diabetes dataset
def diabetes_result(request):
   # Load the model and scaler
    model, scaler = load_model_and_scaler()
 # Get input values from the user
    try:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])
    except ValueError:
        return JsonResponse({"error": "Invalid input. Please enter numeric values only."})

# Scale the input data
    user_input = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    scaled_input = scaler.transform(user_input)

    # Make the prediction
    prediction = model.predict(scaled_input)

    # Render the appropriate result page
    if prediction[0] == 1:
        return render(request, "PositiveDiabetes.html")
    else:
        return render(request, "NegativeDiabetes.html")

    #!return render(request, "predict.html", {"result2": result1})

#!Heart Model-------------------------------------------------------


def heart_result(request):
   
    heart_model, heart_scaler = load_heart_model_and_scaler()
    
    if request.method == 'POST':
        # Get user input from the request using POST
        val1 = float(request.POST.get('age', 0))
        val2 = float(request.POST.get('sex', 0))
        val3 = float(request.POST.get('chest', 0))
        val4 = float(request.POST.get('trestbps', 0))
        val5 = float(request.POST.get('chol', 0))
        val6 = float(request.POST.get('fbs', 0))
        val7 = float(request.POST.get('restecg', 0))
        val8 = float(request.POST.get('thalach', 0))
        val9 = float(request.POST.get('exang', 0))
        val10 = float(request.POST.get('oldpeak', 0))
        val11 = float(request.POST.get('slope', 0))
        val12 = float(request.POST.get('ca', 0))
        val13 = float(request.POST.get('thal', 0))

         # Scale the input values as they need to match the training data scale
        input_data = heart_scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13]])

        # Make a prediction
        pred = heart_model.predict(input_data)

        # Decide the outcome and render the appropriate template
        if pred[0] == 1:
            return render(request, "PositiveHeart.html", {"result2": "Positive for Heart Disease"})  
        else:
            return render(request, "NegativeHeart.html", {"result2": "Negative for Heart Disease"})
    
    # If the request method is not POST, show an error or render a different template
    return render(request, 'error.html', {'message': 'Invalid request method.'})

#!BREAST MODEL------------------------------------------

def breast_result(request):
    # Load the breast model and scaler
    new_breast_model, new_breast_scaler = load_breast_model_and_scaler()

    # Get input from the user
    try:
        val1 = float(request.GET.get('avgtum', 0))
        val2 = float(request.GET.get('texture', 0))
        val3 = float(request.GET.get('peri', 0))
        val4 = float(request.GET.get('area', 0))
        val5 = float(request.GET.get('smooth', 0))
    except ValueError:
        return JsonResponse({"error": "Invalid input. Please enter numeric values only"})

    # Scale the input values to match the model's training data
    input_data = new_breast_scaler.transform([[val1, val2, val3, val4, val5]])

    # Make a prediction
    pred = new_breast_model.predict(input_data)

    # Render the appropriate template based on the prediction
    if pred == [1]:  # Malignant
        return render(request, "malignant.html", {"result2": "Malignant tumor potential"})
    else:  # Benign
        return render(request, "ResultBenign.html", {"result2": "Benign tumor potential"})

#Kidney Logistic Regression
def kidney_result(request):
   
# Load the saved heart disease model and scaler
   # Load the model and scaler
    kidney_model, kidney_scaler = load_kidney_model_and_scaler()
    
    
    # Handle user input (make sure the inputs are valid and match the expected format)
    input_data = pd.DataFrame([[int(request.GET['n1']), float(request.GET['n2']), float(request.GET['n3']),
                                float(request.GET['n4']), float(request.GET['n5']), int(request.GET['n6']),
                                float(request.GET['n7']), int(request.GET['n8']), int(request.GET['n9']),
                                float(request.GET['n10']), float(request.GET['n11']), float(request.GET['n12']),
                                float(request.GET['n13']), float(request.GET['n14']), float(request.GET['n15']),
                                float(request.GET['n16']), float(request.GET['n17']), float(request.GET['n18']),
                                float(request.GET['n19']), int(request.GET['n20']), int(request.GET['n21']),
                                int(request.GET['n22']), int(request.GET['n23']), int(request.GET['n24'])]],)
                               #columns=X.columns)
    

    # Scale the user input
    input_data_scaled = kidney_scaler.transform(input_data)

    # Make the prediction
    pred = kidney_model.predict(input_data_scaled)

    # Render the result
    if pred == 0:
        return render(request, "NegativeCKD.html")
    else:
        return render(request, "PositiveCKD.html")