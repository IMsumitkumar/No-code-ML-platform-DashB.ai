from django.shortcuts import render, redirect
from django.views.generic import TemplateView, CreateView
from . import forms
from django.contrib.auth.decorators import login_required
from django.urls import reverse, reverse_lazy


class UserLogin(TemplateView):
    template_name = 'accounts/auth_login.html'


# Register
class UserRegister(CreateView):
    form_class = forms.UserRegisterForm
    success_url = reverse_lazy('login')
    template_name = 'accounts/auth_register.html'
