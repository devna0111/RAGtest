from django.urls import path
from presentation import views

urlpatterns = [
    path('', views.presentation_analysis, name='presentation_analysis'),
]