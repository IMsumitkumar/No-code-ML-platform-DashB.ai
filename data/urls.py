from data.views import *
from django.urls import path

urlpatterns = [
    path('', dashBoard, name='dashboard'),
    path('data-preprocessing/', Operation, name='data_operation'),
    path('next/', save_processed_dataset, name='next'),
    path('database-dash/', database_dash, name='databasedata'),
]