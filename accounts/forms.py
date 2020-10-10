from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.forms.widgets import PasswordInput, TextInput



class UserRegisterForm(UserCreationForm):
	email = forms.EmailField()

	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']


class CustomAuthForm(AuthenticationForm):
	username = forms.CharField(widget=TextInput(attrs={}))
	password = forms.CharField(widget=PasswordInput(attrs={}))