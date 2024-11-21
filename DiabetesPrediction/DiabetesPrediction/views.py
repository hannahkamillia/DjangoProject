from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    data = pd.read_csv(r"C:\Users\Nur Athirah\Downloads\Heart_Disease_Prediction.csv")

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
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\Breast_cancer_data.csv")

 # Prepare features (X) and target (y)
    X = data.drop(columns=['diagnosis'])  # Drop the 'diagnosis' column for input features
    y = data['diagnosis']  # 'diagnosis' column is the target variable


# Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

 # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

# Initialize and train the Logistic Regression model
    breast_model = LogisticRegression(random_state=42, max_iter=1000)
    breast_model.fit(X_train, y_train)

    
 # Get user input from the request
    val1 = float(request.GET['avgtum'])
    val2 = float(request.GET['texture'])
    val3 = float(request.GET['peri'])
    val4 = float(request.GET['area'])
    val5 = float(request.GET['smooth'])

# Scale the input values as they need to match the training data scale
    input_data = scaler.transform([[val1, val2, val3, val4, val5]])

# Make a prediction
    pred = breast_model.predict(input_data)

# Decide the outcome and render the appropriate template
    if pred == [1]:
        return render(request, "malignant.html", {"result2": "Malignant tumor potential"})  
    else:
        return render(request, "negative.html", {"result2": "Negative for Heart Disease"})
    
#Kidney Model
def kidney_result(request):

    #Load and preprocess the data
    data = pd.read_csv(r"C:\Users\Nur Athirah\Downloads\kidney_disease.csv")

    #Encode categorical features
    label_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    for col in label_cols:
        data[col] = data[col].astype(str)
        data[col] = data[col].factorize()[0]

    #Replace blank values with NaN and handle missing values
    #data.replace('\t?', pd.NA, inplace=True)
    #data.fillna(data.mean(), inplace=True)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    #Separate features and target
    X = data.drop(columns=["classification"])
    y = data["classification"]

    #Replace non-numeric values with NaN, then handle NaNs (e.g., fill with mean)
    X = X.apply(pd.to_numeric, errors='coerce')

    #Fill NaNs with the mean of each column
    X.fillna(X.mean(), inplace=True)

    # Ensure y is also numeric, if needed
    y = pd.to_numeric(y, errors='coerce')

    # Train a temporary Random Forest to get feature importances
    rf_temp = RandomForestClassifier(random_state=42)
    rf_temp.fit(X, y)

    # Get feature importances and select the top features
    feature_importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(8).index  # Select top 8 features
    X_top = X[top_features]

    # Split the data using only the top features
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=101)

    # Train the final model using only the selected features
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Testing function in Django
    # def kidney_result(request):
    # Input from user based on selected top features
    kid_val1 = float(request.GET['n1'])
    kid_val2 = float(request.GET['n2'])
    kid_val3 = float(request.GET['n3'])
    kid_val4 = float(request.GET['n4'])
    kid_val5 = float(request.GET['n5'])
    kid_val6 = float(request.GET['n6'])
    kid_val7 = float(request.GET['n7'])
    kid_val8 = float(request.GET['n8'])

    # Predict based on user input
    pred = model.predict([[kid_val1, kid_val2, kid_val3, kid_val4, kid_val5, kid_val6, kid_val7, kid_val8]])

    # Show the result
    if pred == [1]:
        return render(request, "positive.html", {"result2": "Positive for Kidney Disease"})  
    else:
        return render(request, "negative.html", {"result2": "Negative for Kidney Disease"})
