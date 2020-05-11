import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import joblib
from langdetect import detect
import urllib.request
import urllib.parse
from selenium import webdriver
from urllib.request import Request, urlopen
from flashtext.keyword import KeywordProcessor
import re
import ssl
import requests
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from inscriptis import get_text
import pickle
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import joblib



url_list = []
user_input = input('Please Enter Url (eg :"website.com")')
url_list.append(user_input)
print(url_list)

empty=[]
def scrapingdata(urlfiles):
    list_url = []
    for url in urlfiles:

        try:
            final_url = 'http://' + url
            r = requests.get(final_url, allow_redirects=True)
            fullurl = r.url
            protocol = fullurl[:fullurl.find(":")]
            print(protocol)

            ## Use either requests or selenium to know whether website is http or https

            # driver = webdriver.Chrome('/home/user/PycharmProjects/seleniumScrape/chromedriver')
            # driver.get(final_url)
            # urll = driver.current_url
            # url_type = urll[:urll.find(":")]
            # print(url_type)
            # print(type(url_type))
            # driver.quit()


            if protocol == 'http':
                found_url_type = 'http://' + url
                list_url.append(found_url_type)
            elif protocol == 'https':
                found_url_type = 'https://' + url
                list_url.append(found_url_type)

            else:
                found_url_type = 'http://' + url
                list_url.append(found_url_type)
        except:
            pass

    #print(list_url)
    #start = 0
    for all_list in list_url:
        print(type(all_list),'all list type')
        try:
            # html = urllib.request.urlopen(str(all_list)).read().decode('utf-8')

            ## Adding user agent to hide bot access
            ssl.match_hostname = lambda cert, hostname: True
            new_url = all_list
            req = Request(new_url,headers={'User-Agent':'Mozilla/5.0'})
            html = urlopen(req).read().decode('utf-8')
            text = get_text(html)
            language = detect(text)
            print(language)

            ## If condition matches to English then data preprocessing starts
            if language == 'en':
                #English_url.append(all_list)
                data = text.split()
                no_punc = [char for char in data if char not in string.punctuation]

                ## Stripping data
                no_punc1 = [stripp.strip('?#{}()@%£€"[]!.\'\",:;_▼   ™®©+$-*/&|<>=~0123456789') for stripp in no_punc]
                more_words = [word1 for word1 in no_punc1 if len(no_punc1) > 2]
                res = []
                [res.append(x) for x in more_words if x not in res]
                # no_punc = ''.join(no_punc)

                ## Importing stopwords and removing from extracted data
                ## Change path of text file
                newStopWords = pd.read_csv("/home/user/Downloads/stop.txt")
                extended_stopwords = ''.join(newStopWords['a'].tolist())
                clean_words = [word for word in res if word.lower() not in extended_stopwords]
                # stripped_data = [stripp for stripp in clean_words
                # if  clean_words.strip('{}()"[]!.\'\",:;+$-*/&|<>=~0123456789')]
                lowered = [lowercase.lower() for lowercase in clean_words]
                # new_data = [data.strip('{}()"[]!.\'\",:;+$-*/&|<>=~0123456789') for data in lowered]
                ## Removing digits from text
                no_integers = [x for x in lowered if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
                ## Creating text file for each English url data and storing in new folder (folder needs to create manually)
                ##start = start + 1
                # save_path = '/home/user/Desktop/temp' #*********1/2 path need to change
                # ##new_file = 'start' + str(start) + '.txt'
                # a_string = all_list
                # split_string = a_string.split("//", 1)
                #
                # substring = split_string[1]
                # print(substring,'substring is here')
                # print(type(substring),'new_file type is here')
                # new_file = substring + '.txt'
                # completeName = os.path.join(save_path,new_file)
                # with open(completeName, 'w') as f:
                #     file = f.write(' '.join(no_integers))

                empty.append(no_integers)
                print("completing each loop")


            else:
                print('Non English Url')
                #Non_English_url.append(all_list)




        except:
            pass
try:
    print(scrapingdata(url_list))

    df = pd.DataFrame(empty)
    #df['Data'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['Data'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df1 = df['Data'].tolist()
    print(df1,'df is here')

    model,tfidf = pickle.load(open('MultiLiModel.pkl','rb'))
    sample1 = tfidf.transform(df1)
    pred = model.predict(sample1)

    if pred == 0:
        print('Alcohol')
    elif pred == 1:
        print('War-violence-weapon')
    elif pred == 2:
        print('Religion')
    elif pred == 3:
        print('Porn')
    elif pred == 4:
        print('Other Category')
    else:
        print("Something is wrong please check")

except ValueError:
    print("Something is wrong with your url please try with another url.")
    print("Also check:")
    print("1.As the website has more text or not")
else:
    print("Everything went successfully.")






# ## Importing scraped saved text files data from folder
# ## Change loc path
# loc = '/home/user/Desktop/temp' #*********2/2 path need to change
# os.chdir(loc)
# filelist = os.listdir()
# # print (len((pd.concat([pd.read_csv(item, names=[item[:-4]]) for item in filelist],axis=1))))
#
# data = []
# path = loc
# files = [f for f in os.listdir(path) if os.path.isfile(f)]
# for f in files:
#     with open(f, 'r') as myfile:
#         data.append(myfile.read())
#
# df = pd.DataFrame(data, columns=['Data'])
# print(df.shape)
# new_df = df.loc[0:45]
# # new_df['Label'] = 'alcohol'
# new_df.insert(1, 'Label', 'unknown')
# print(new_df.shape)
# print(new_df.head())
# X_test = new_df['Data']




