from django.urls import path
from . import views

urlpatterns = [
    path('', views.presentation_analysis, name='presentation_analysis'),
]