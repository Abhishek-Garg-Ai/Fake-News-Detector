from flask import Flask, render_template, request, jsonify
import nltk
import pandas as pd
import numpy as np
import re
from googlesearch import search
from tqdm import tqdm
import spacy

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
nltk.download('punkt')

import pickle
from newspaper import Article
def get_data(url):
  article=Article(url)
  try:
    article.download()
  except Exception as e:
    return False,'',[],''
  try:
    article.parse()
    article.nlp()
    title=article.title
    keywords=article.keywords
    summary=article.summary
    if(len(title)<=5 or len(keywords)==0 or len(summary)<=5):
      return False,'',[],''
    else:
      return True,title,keywords,summary
  except Exception as e:
      return False,'',[],''
  print(e)
# from sentence_transformers import SentenceTransformer,util

app = Flask(__name__)

model = pickle.load(open('random_model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))
# cosmodel = pickle.load(open('cossmodel.pkl', 'rb')).to('cpu')
nlp=spacy.load('en_core_web_sm')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    text=text.lower()
    test_list = search(text, tld='co.in', num=10, stop=10, pause=2)
    useless_domain = (
    'https://en.wikipedia', 'https://www.linkedin', 'https://in.linkedin', 'https://expertportals.com',
    'https://academictree.org', 'https://facebook.com', 'https://youtube.com', 'https://m.facebook.com',
    'https://www.facebook.com')

    results = filter(lambda x: not x.startswith(useless_domain), test_list)
    title_lt = []
    keyword_lt = []
    summary_lt = []
    for url in tqdm(results):
        reachable, title, keywords, summary = get_data(url)
        if (reachable):
            title_lt.append(title)
            keyword_lt.append(keywords)
            summary_lt.append(summary)
    test_df = pd.DataFrame({"summary": summary_lt, "title": title_lt})
    test_df['summary'].str.lower()
    test_df['summary'].apply(cleanHtml)
    test_df['summary'].apply(cleanPunc)
    test_df['summary'].apply(keepAlpha)
    vectorized_test = tfidfvect.transform(test_df['summary'])

    test_df['Random_prob'] = model.predict_proba(vectorized_test)[:,0]
    Av_random=np.average(test_df['Random_prob'])
    Av_random


    text1=nlp(text)
    cosine_scores=[]
    for summary in summary_lt:
        text2=nlp(summary)
        cosine_scores.append(text1.similarity(text2))
    #test_df['sim']=cosine_scores
    #print(test_df)

    def Average(cosine_scores):
        return sum(cosine_scores) / len(cosine_scores)

    average = Average(cosine_scores)
    #test_df['similarity']=cosine_scores
    #print(test_df)


    prediction = 'Prediction of this news 📰 is REAL 🧐 ' if ((Av_random) >=0.5 and average >= 0.5) else 'Prediction for this  news 📰 is FAKE 🧐'
    if(text.find("not")!=-1):
        prediction = 'Prediction of this news 📰 is FAKE 🧐 ' if ((Av_random) >=0.5 and average >= 0.5) else 'Prediction for this  news 📰 is REAL 🧐'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
