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
      return render(request, "NegativeDiabetes.html", {"result2": result1})  
    else:
        return render(request, "PositiveDiabetes.html", {"result2": result1})

    #!return render(request, "predict.html", {"result2": result1})

#!Heart Model-------------------------------------------------------
def heart_result(request):
    # Load the heart disease dataset
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\Heart_Disease_Prediction.csv")

    # Check for missing values
    print(data.isnull().sum())

    # Convert categorical variables to numerical
    data['Sex'] = data['Sex'].map({0: 0, 1: 1})  # Already numerical
    data['Heart Disease'] = data['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # Select features and target variable
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    heart_model.fit(X_train, y_train)

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
        input_data = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13]])

        # Make a prediction
        pred = heart_model.predict(input_data)

        # Decide the outcome and render the appropriate template
        if pred == [1]:
            return render(request, "PositiveHeart.html", {"result2": "Positive for Heart Disease"})  
        else:
            return render(request, "NegativeHeart.html", {"result2": "Negative for Heart Disease"})
    
    # If the request method is not POST, show an error or render a different template
    return render(request, 'error.html', {'message': 'Invalid request method.'})

#!BREAST MODEL------------------------------------------
def breast_result(request):

#load data breast dataset
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\Breast_cancer_data.csv")
    data.dropna(inplace=True)  # Remove rows with missing values

    # Features and target variable
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    breast_model = RandomForestClassifier(n_estimators=100, random_state=42)
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
        return render(request, "ResultBenign.html", {"result2": "Negative for Heart Disease"})

#Kidney Logistic Regression
def kidney_result(request):
    data = pd.read_csv(r"C:\Users\Hannah Kamillia\Downloads\kidney_disease.csv")

    # Drop ID column
    data = data.drop(columns=["id"], errors='ignore')

    #Map categorical columns to binary values
    binary_mapping = {"yes": 1, "no": 0,
                "good": 1, "poor": 0,
                "present": 1, "notpresent": 0,
                "normal": 1, "abnormal": 0}

    # Handle missing values by imputing with the mean (for numerical columns)
    imputer = SimpleImputer(strategy='mean')  # Can change strategy if needed
    data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=['float64', 'int64'])), columns=data.select_dtypes(include=['float64', 'int64']).columns)
    
    # Merge the cleaned data back
    data[data_imputed.columns] = data_imputed

    # Handle missing categorical data by using the mode (most frequent value)
    cat_columns = data.select_dtypes(include=['object']).columns
    for col in cat_columns:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

    # Apply mapping to all columns except 'classification'
    for col in data.columns:
        if col != 'classification' and data[col].dtype == 'object':  # Only apply this to string columns, excluding 'classification'
            # Replace categorical values based on the binary_mapping
            data[col] = data[col].replace(binary_mapping)
        
            # Only apply .str methods if the column is truly of string type
            if data[col].dtype == 'object':  # Check if the column is still of string type
                data[col] = data[col].str.strip()  # Remove leading/trailing spaces
                data[col] = data[col].replace(r'\t\?', '', regex=True)  # Remove the specific '\t?' if exists
        
            # Convert to numeric, and fill NaN values with 0 (or another suitable value)
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, NaN for invalid
            data[col] = data[col].fillna(0)  # Fill NaNs with 0
            data[col] = data[col].astype(int)  # Ensure the result is integer (1 or 0)

    # Now map the 'classification' column to binary values manually
    data['classification'] = data['classification'].replace({"ckd": 1, "ckd\t":1, "notckd": 0})

    # Separate features and target
    X = data.drop("classification", axis=1)
    Y = data['classification']

    #Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scale the features (ensure all values are numeric and clean)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, Y_train)

    # Initialize and train the model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, Y_train)

    # Handle user input (make sure the inputs are valid and match the expected format)
    input_data = pd.DataFrame([[int(request.GET['n1']), float(request.GET['n2']), float(request.GET['n3']),
                                float(request.GET['n4']), float(request.GET['n5']), int(request.GET['n6']),
                                int(request.GET['n7']), int(request.GET['n8']), int(request.GET['n9']),
                                float(request.GET['n10']), float(request.GET['n11']), float(request.GET['n12']),
                                float(request.GET['n13']), float(request.GET['n14']), float(request.GET['n15']),
                                float(request.GET['n16']), float(request.GET['n17']), float(request.GET['n18']),
                                int(request.GET['n19']), int(request.GET['n20']), int(request.GET['n21']),
                                int(request.GET['n22']), int(request.GET['n23']), int(request.GET['n24'])]],
                               columns=X.columns)
    

    # Scale the user input
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    pred = model.predict(input_data_scaled)

    # Render the result
    if pred == 0:
        return render(request, "NegativeCKD.html")
    else:
        return render(request, "PositiveCKD.html")