from django.contrib import admin
from django.urls import path, include
from data.views import upload_data
from django.contrib.auth import views as auth_views
from django.conf import settings
from accounts import views as acc_views
from accounts.forms import CustomAuthForm
from django.conf.urls.static import static
from .views import index

urlpatterns = (
    path("admin/", admin.site.urls),
    path('', index, name='home'),

    path('upload_dataset/', upload_data, name='upload'),

    path('django_plotly_dash/', include('django_plotly_dash.urls')), 
    
    path("login/",
         auth_views.LoginView.as_view(template_name="accounts/auth_login.html", authentication_form=CustomAuthForm),
         name='login'),
    path('register/', acc_views.UserRegister.as_view(), name='register'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='accounts/password_reset.html'),
         name='password_reset'),
    path('password-reset/done/',
         auth_views.PasswordResetDoneView.as_view(template_name='accounts/password_reset_done.html'),
         name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(template_name='accounts/password_reset_confirm.html'),
         name='password_reset_confirm'),
    path('password-reset-complete/',
         auth_views.PasswordResetCompleteView.as_view(template_name='accounts/password_reset_complete.html'),
         name='password_reset_complete'),

    path('dash/', include("data.urls")),
    path('viz/', include("Viz.urls")),
)

