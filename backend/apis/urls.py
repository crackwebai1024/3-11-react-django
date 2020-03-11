from django.urls import path, include
from .views import SentimentAnalysis, ArticleRecommender, LanguageDetection, TextSummarization
urlpatterns = [
    path('SentimentAnalysis/', SentimentAnalysis.as_view()),
    path('ArticleRecommender/', ArticleRecommender.as_view()),
    path('LanguageDetection/', LanguageDetection.as_view()),
    path('TextSummarization/', TextSummarization.as_view()),
]