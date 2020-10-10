from data.views import *
from django.urls import path

urlpatterns = [
    path('', dashBoard, name='dashboard'),
    path('data_opt/', Operation, name='data_operation'),
]