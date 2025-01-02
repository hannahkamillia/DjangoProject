"""
URL configuration for DiabetesPrediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

from django.contrib import admin
from django.urls import path
from . import views  # Ensure you're importing views correctly

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Home and Menu routes
    path('', views.home, name='home'),
    path('menu/', views.menu),
    path('menu/home', views.home),
    path('menu/menu', views.menu),
    path('menu/menu/kidney', views.kidney),
    

    # Prediction routes
    path('predict/', views.predict),
    path('menu/diabetes_result', views.diabetes_result),
    path('predict/diabetes_result', views.diabetes_result),  # Updated from 'result' to 'diabetes_result'
    path('menu/predict', views.predict),
    path('menu/heart', views.heart),
    path('menu/menu/predict', views.predict),
    path('menu/heart_result', views.heart_result),  # Updated from 'result' to 'diabetes_result'
    path('menu/breast', views.breast),
    path('menu/menu/breast', views.breast),
    path('menu/breast_result', views.breast_result),
    path('breast/breast_result', views.breast_result),
    path('menu/kidney', views.kidney),
    path('menu/menu/kidney', views.kidney),
    path('menu/kidney_result', views.kidney_result),
    path('kidney/kidney_result', views.kidney_result),
    path('heart_result/', views.heart_result, name='heart_result'),

    #Treatment Pages
    path('menu/DiabetesTreat', views.DiabetesTreat),
    path('predict/DiabetesTreat', views.DiabetesTreat),
    path('menu/HeartTreat', views.HeartTreat),
    path('heart/HeartTreat', views.HeartTreat),
    path('menu/BreastTreat', views.BreastTreat),
    path('menu/KidneyTreat', views.KidneyTreat),


    # Positive/Negative pages
    path('predict/NegativeDiabetes', views.NegativeDiabetes),
    path('predict/PositiveDiabetes', views.PositiveDiabetes),
    path('kidney/NegativeCKD', views.NegativeCKD),
    path('kidney/PositiveCKD', views.PositiveCKD),
    path('heart/NegativeHeart', views.NegativeHeart),
    path('heart/PositiveHeart', views.PositiveHeart),
    path('breast/ResultBenign', views.ResultBenign),
    path('breast/Malignant', views.malignant),

    # Recommendation page
    path('predict/Recomm', views.Recomm),
    path('menu/DiabetesTreat', views.DiabetesTreat),
    path('menu/HeartTreat', views.HeartTreat),
    path('menu/KidneyTreat', views.KidneyTreat),
    path('menu/BreastTreat', views.BreastTreat),

    # Back to Home
    path('heart_result/HeartTreat', views.HeartTreat),
    path('breast_result/home', views.home),
    path('kidney_result/home', views.home),
    path('diabetes_result/home', views.home),
    path('heart_result/home', views.home),

    # Heart model prediction routes
    path('predict/heart', views.heart_result),  # Updated to 'heart_result'
    path('menu/heart', views.heart_result),  # Updated to 'heart_result'

    # Breast model prediction routes
    path('predict/breast', views.breast_result),  # Updated to 'breast_result'
    path('menu/breast', views.breast_result),  # Updated to 'breast_result'

    #From result to menu
    path('breast_result/menu', views.menu),
    path('heart_result/menu', views.menu),
    path('kidney_result/menu', views.menu),
    path('diabetes_result/menu', views.menu),

]

