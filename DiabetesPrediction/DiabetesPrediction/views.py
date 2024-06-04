from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

#!Read the data from the dataset
def result(request):
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\diabetes.csv")

#!Train test split
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#!Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

#!Input from user
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

#!Decide to show the outcome
    result1 = ""
    if pred==[1]:
      return render(request, "positive.html", {"result2": result1})  
    else:
        return render(request, "negative.html", {"result2": result1})

    #!return render(request, "predict.html", {"result2": result1})

