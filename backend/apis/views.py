from django.shortcuts import render
from .models import History
# Create your views here.
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.generics import ListCreateAPIView, CreateAPIView
from rest_framework.response import Response
from django.views.decorators.csrf import ensure_csrf_cookie
from .serializers import UserHistorySerializer

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
from datetime import date

nlp = spacy.load('en_core_web_sm')
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

def input(text, action, token):
    history = History()
    history.token = token
    history.action = action
    history.text = text
    today = date.today()
    history.time = today.strftime("%Y-%m-%d")
    history.save()

class SeeUserHistory(APIView):
    def get(self, request):
        token = self.request.query_params.get('token')
        data = History.objects.filter(token = token)
        length = len(data)
        action = []
        text = []
        time = []
        for i in range(length):
            x = str(data[i])
            part = x.split("{{{{{***}}}}}")
            action.append(part[0])
            text.append(part[1])
            time.append(part[2])
        response_data = { "action": action, "text":text, "time": time }
        return Response(response_data)
        
class SentimentAnalysis(APIView):
    def __init__(self, request):
        text = self.request.query_params.get('text')
        action = "Sentiment Analysis"
        token = self.request.query_params.get('token')
        input(text, action, token)
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(text)       
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#F7464A', '#46BFBD', '#FDB45C']
        values = [sentiment_dict['pos'], sentiment_dict['neg'], sentiment_dict['neu']]
        response_data={"labels": labels, "colors": colors, "values": values}
        return Response(response_data)

class ArticleRecommender(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Article Recommender"
        token = self.request.query_params.get('token')
        input(text, action, token)
        df = pd.read_csv("apis/data/keyworsemerj1.csv") 
        int_features = [x for x in self.request.query_params.get('text')]
        results = df[df["KEYWORDS"].str.contains(int_features[0])] 
        url_title = results.drop(["KEYWORDS", "CONTENT"], axis=1)
        response_data={"tables": url_title.values}
        return Response(response_data)

class LanguageDetection(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Language Detection"
        token = self.request.query_params.get('token')
        input(text, action, token)
        results = detect(text)
        response_data = {"rawtext": results}
        return Response(response_data)

class TextSummarization(APIView):
    def tokenizer(self, s):
        tokens = []
        for word in s.split(' '):
            tokens.append(word.strip().lower())
        return tokens

    def sent_tokenizer(self, s):
        sents = []
        for sent in s.split('.'):
            sents.append(sent.strip())
        return sents

    def count_words(self,tokens):
        word_counts = {}
        for token in tokens:
            if token not in stop_words and token not in punctuation:
                if token not in word_counts.keys():
                    word_counts[token] = 1
                else:
                    word_counts[token] += 1
        return word_counts

    def word_freq_distribution(self, word_counts):
        freq_dist = {}
        max_freq = max(word_counts.values())
        for word in word_counts.keys():  
            freq_dist[word] = (word_counts[word]/max_freq) #divide the number of occurrences of each word by the number of occurrences of the word which occurs most in the document
        return freq_dist

    def score_sentences(self, sents, freq_dist, max_len=40):
        sent_scores = {}  
        for sent in sents:
            words = sent.split(' ')
            for word in words:
                if word.lower() in freq_dist.keys():
                    if len(words) < max_len:
                        if sent not in sent_scores.keys():
                            sent_scores[sent] = freq_dist[word.lower()]
                        else:
                            sent_scores[sent] += freq_dist[word.lower()]
        return sent_scores

#sent_scores = score_sentences(sents, freq_dist)
#sent_scores

# select the top k sentences to represent the summary of the article.

    def summarize(self, sent_scores, k):
        top_sents = Counter(sent_scores) 
        summary = ''
        scores = []
        
        top = top_sents.most_common(k)
        for t in top: 
            summary += t[0].strip()+'. '
            scores.append((t[1], t[0]))
        return summary[:-1], scores

# call the function to generate the summary
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Text Summarization"
        token = self.request.query_params.get('token')
        input(text, action, token)
        #######
        tokens = self.tokenizer(text)
        sents = self.sent_tokenizer(text)
        word_counts = self.count_words(tokens)
        freq_dist = self.word_freq_distribution(word_counts)
        sent_scores = self.score_sentences(sents, freq_dist)
        summary, summary_sent_scores = self.summarize(sent_scores, 5)
        datatextsum = [summary_sent_scores,summary,text]
        response_data = {"textdata": datatextsum}
        #########
        return Response(response_data)

class EntityExtraction(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Entity Extraction"
        token = self.request.query_params.get('token')
        input(text, action, token)
        docx = nlp(text)
        html = displacy.render(docx,style="ent")
        html = html.replace("\n\n","\n")
        result = HTML_WRAPPER.format(html)
        response_data = {"rawtext": text, "result": result}
        return Response(response_data)

class KeywordExtraction(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Keyword Extraction"
        token = self.request.query_params.get('token')
        input(text, action, token)

        # Reka setup with stopword directory
        stop_dir = "apis/data/a.txt"
        rake_object = RAKE.Rake(stop_dir)

        # Extract keywords
        keywords = rake_object.run(text)
        response_data = {"keywords": keywords}
        return Response(response_data)

class ArticleExtraction(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Article Extraction"
        token = self.request.query_params.get('token')
        input(text, action, token)
        url = [x for x in self.request.query_params.get('text')]
        linksdata = url[0]
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        res = requests.get(linksdata, headers=headers)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        texts = soup.find_all('p')
        data = ''
        NEWLINE = '\n'
        for p in texts:
            p = p.text
            data += p
            data += NEWLINE
        tokens = tokenizer(data)
        sents = sent_tokenizer(data)
        word_counts = count_words(tokens)
        freq_dist = word_freq_distribution(word_counts)
        sent_scores = score_sentences(sents, freq_dist)
        summary, summary_sent_scores = summarize(sent_scores, 5)
        datatextsum = [summary,data]
        return render_template('articleextract.html',textdata = datatextsum)

class Tokenize(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Tokenize"
        token = self.request.query_params.get('token')
        input(text, action, token)
        tokens = word_tokenize(text)
        text = [i for i in tokens]
        response_data = {"result": text}
        return Response(response_data)

class ConvertUppercasetoLower(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Convert Uppercase to Lower"
        token = self.request.query_params.get('token')
        input(text, action, token)
        input_str = text.lower()
        response_data = {"result": input_str}
        return Response(response_data)

class RemoveNumbers(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Remove Numbers"
        token = self.request.query_params.get('token')
        input(text, action, token)
        result = re.sub(r"\d+", "", text)
        response_data = {"result": result}
        return Response(response_data)

class RemoveWhiteSpaces(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Remove WhiteSpaces"
        token = self.request.query_params.get('token')
        input(text, action, token)
        result = text.replace(" ", "")
        response_data = {"result": result}
        return Response(response_data)

class RemoveStopwords(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Remove Stopwords"
        token = self.request.query_params.get('token')
        input(text, action, token)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text) 
        text = [w for w in word_tokens if not w in stop_words] 
        text = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                text.append(w)
        response_data = {"result": text}
        return Response(response_data)

class Plagarism(APIView):
    def get(self, request):
        text = self.request.query_params.get('text')
        action = "Plagarism"
        token = self.request.query_params.get('token')
        input(text, action, token)
        response_data = {"result": text}
        return Response(response_data)

