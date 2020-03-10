from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from django.views.decorators.csrf import ensure_csrf_cookie


#this is for api#
import numpy as np
from collections import Counter 
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stop_words
import pickle
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import re
import spacy
import string
from spacy import displacy
from langdetect import detect
from langdetect import detect_langs
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import RAKE
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from bokeh.io import show, output_file
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
import nltk
from nltk.stem import 	WordNetLemmatizer
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #to convert strings to numerical vectors
from nltk.corpus import stopwords

import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
### machine learning module import ###

from django.http import HttpResponse, JsonResponse
class SentimentAnalysis(APIView):
    def get(self, request):
        print(request)
        text = self.request.query_params.get('text')
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(text)       
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#F7464A', '#46BFBD', '#FDB45C']
        values = [sentiment_dict['pos'], sentiment_dict['neg'], sentiment_dict['neu']]
        print(labels, colors, values)
        response_data={"labels": labels, "colors": colors, "values": values}
        return Response(response_data)