from .models import DataSet, ProcessedDataSet
from django import forms


class DataSetForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = '__all__'
        exclude = ['user']


class ProcessedDatasetForm(forms.ModelForm):
    class Meta:
        model = ProcessedDataSet
        fields = '__all__'
        exclude = ['user']