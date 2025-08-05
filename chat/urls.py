from django.urls import path
from chat import views

urlpatterns = [
    path('', views.chat_page, name='chat_page'),
    path('upload/', views.file_upload, name='file_upload'),
    path('chat/send/', views.chat_send, name='chat_send')
]