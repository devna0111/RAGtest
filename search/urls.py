from django.urls import path
from search import views

app_name = "search"

urlpatterns = [
    path('', views.search_page, name='search_page'),
    path('result/', views.result, name="result")
]