from django.urls import path
from .views import SentimentAnalysis
urlpatterns = [
    path('api/SentimentAnalysis/', SentimentAnalysis.as_view()),
    path('', SentimentAnalysis.as_view()),
]