from django.urls import path, include
from rest_framework import routers
from .views import *

urlpatterns = [
    path('SentimentAnalysis/', SentimentAnalysis.as_view()),
    path('ArticleRecommender/', ArticleRecommender.as_view()),
    path('LanguageDetection/', LanguageDetection.as_view()),
    path('TextSummarization/', TextSummarization.as_view()),
    path('EntityExtraction/', EntityExtraction.as_view()),
    path('KeywordExtraction/', KeywordExtraction.as_view()),
    path('ArticleExtraction/', ArticleExtraction.as_view()),
    path('Tokenize/', Tokenize.as_view()),
    path('ConvertUppercasetoLower/', ConvertUppercasetoLower.as_view()),
    path('RemoveNumbers/', RemoveNumbers.as_view()),
    path('RemoveWhiteSpaces/', RemoveWhiteSpaces.as_view()),
    path('RemoveStopwords/', RemoveStopwords.as_view()),
    path('Plagarism/', Plagarism.as_view()),
    path('seehistory/', SeeUserHistory.as_view()),
]