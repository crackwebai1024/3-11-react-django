from django.urls import path
from .views import SentimentAnalysis, ArticleRecommender
urlpatterns = [
    path('api/SentimentAnalysis/', SentimentAnalysis.as_view()),
    path('api/ArticleRecommender/', ArticleRecommender.as_view()),
    path('', SentimentAnalysis.as_view()),
]