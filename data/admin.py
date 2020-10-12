from django.contrib import admin
from .models import DataSet, ProcessedDataSet


# Register your models here.
admin.site.register(DataSet)
admin.site.register(ProcessedDataSet)