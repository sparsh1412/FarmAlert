from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
	path('',views.serve,name = 'farm-services'),
	path('govt_alerts/', views.govt_alert, name = 'govt-alerts'),
    path('cold_storages_near_me/', views.cold_storages, name = 'cold-storages'),
    path('crop_recommendation/',views.CropRecommender,name = 'crop-recommender'),
    path('recommendation/',views.Recommendation,name = 'crop-recommendation'),
    path('yield_pred/',views.yieldPred,name = 'yield-predictor'),
]
