from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_page, name='chat_page'),
    path('upload/', views.file_upload, name='file_upload'),
]