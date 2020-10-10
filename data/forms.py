from .models import DataSet
from django import forms


class DataSetForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = '__all__'
        exclude = ['user']
