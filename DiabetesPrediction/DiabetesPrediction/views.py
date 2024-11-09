from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def home(request):
    return render(request, 'home.html')

def menu(request):
    return render(request, 'menu.html')

def predict(request):
    return render(request, 'predict.html')

def negative(request):
    return render(request, 'negative.html')

def positive(request):
    return render(request, 'positive.html')

def Recomm(request):
    return render(request, 'Recomm.html')

def heart(request):
    return render(request, 'heart.html')

def breast(request):
    return render(request, 'breast.html')

#!Read the data from the diabetes dataset
def diabetes_result(request):
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\diabetes.csv")

#!Train test split
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


#!Train the diabetes model
    diabetes_model = LogisticRegression()
    diabetes_model.fit(X_train, Y_train)

#!Input from user
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = diabetes_model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

#!Decide to show the outcome
    result1 = ""
    if pred==[1]:
      return render(request, "positive.html", {"result2": result1})  
    else:
        return render(request, "negative.html", {"result2": result1})

    #!return render(request, "predict.html", {"result2": result1})

#!Heart Model
def heart_result(request):
    # Load the heart disease dataset
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\Heart_Disease_Prediction.csv")

    # Prepare features (X) and target (y)
    X = data.drop(columns=['Heart Disease'])  
    y = data['Heart Disease']  

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_scaled, y_train)  # You need to fit the model before making predictions

    # Get user input from the request
    val1 = float(request.GET['age'])
    val2 = float(request.GET['sex'])
    val3 = float(request.GET['chest'])
    val4 = float(request.GET['trestbps'])
    val5 = float(request.GET['chol'])
    val6 = float(request.GET['fbs'])
    val7 = float(request.GET['restecg'])
    val8 = float(request.GET['thalach'])
    val9 = float(request.GET['exang'])
    val10 = float(request.GET['oldpeak'])
    val11 = float(request.GET['slope'])
    val12 = float(request.GET['ca'])
    val13 = float(request.GET['thal'])

    # Scale the input values as they need to match the training data scale
    input_data = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13]])

    # Make a prediction
    pred = logistic_model.predict(input_data)

    # Decide the outcome and render the appropriate template
    if pred == [1]:
        return render(request, "positive.html", {"result2": "Positive for Heart Disease"})  
    else:
        return render(request, "negative.html", {"result2": "Negative for Heart Disease"})

#!Breast Model
def breast_result(request):

#load data breast dataset
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\Breast_Cancer.csv")

 # Prepare features (X) and target (y)
    X = data[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive']].copy()
    y = data['Status']

#split traintest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

# Initialize and train the logistic regression model
    breast_model = LogisticRegression(random_state=42)
    breast_model.fit(X_train_scaled, y_train)
    
 # Get user input from the request
    val1 = float(request.GET['age'])
    val2 = float(request.GET['size'])
    val3 = float(request.GET['node'])
    val4 = float(request.GET['nodpos'])

# Scale the input values as they need to match the training data scale
    input_data = scaler.transform([[val1, val2, val3, val4]])

# Make a prediction
    pred = breast_model.predict(input_data)

# Decide the outcome and render the appropriate template
    if pred == [1]:
        return render(request, "positive.html", {"result2": "Positive for Heart Disease"})  
    else:
        return render(request, "negative.html", {"result2": "Negative for Heart Disease"})






