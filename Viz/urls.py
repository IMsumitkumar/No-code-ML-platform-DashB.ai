from Viz.views import *
from django.urls import path

urlpatterns = [
    path('', visual, name='visual'),
]