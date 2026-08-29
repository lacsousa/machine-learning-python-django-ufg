from django.contrib import admin
from django.urls import path
from . import views

app_name = 'app_am'

urlpatterns = [
    path('', views.am_exemplo_01, name='am_exemplo_01'),
    path('am_exemplo_01', views.am_exemplo_01, name='am_exemplo_01'),
]