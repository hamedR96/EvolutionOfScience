import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from top2vec import Top2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import re
import nltk

df=pd.read_csv('arxiv_CS_1998_2017.csv',sep=';',low_memory=False)

df.dropna(subset = ["item3"], inplace=True)
df.dropna(subset = ["item6"], inplace=True)


for i in df.index:
    if not df['item3'][i].isnumeric() :
        df.drop(i, inplace=True)

year_list=df['item3'].tolist()

first_year=int(min(year_list))
last_year=int(max(year_list))
number_years=last_year-first_year+1
#print(number_years)

list= [[] for _ in range(20)]

for i in df.index:
    index=int(df['item3'][i])-first_year
    list[index].append(df['item6'][i])

def text_cleanner(list):
    text_clean = [re.sub(r'http\S+', '', t) for t in list]
    text_clean= [t.strip().replace('\n', ' ') for t in text_clean]
    docs = [re.sub(r'[\w\.-]+@[\w\.-]+', '', t) for t in text_clean]
    docs = [t.strip().replace('`', ' ') for t in docs]
    return [t.strip().replace('--', ' ') for t in docs]
stemmer = nltk.stem.snowball.SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
def lemmatizering(list):
    preproccessed_doc=[]
    for doc in list:
        doc=doc.replace('.', " ")
        doc = doc.replace(',', " ")
        doc = doc.replace(';', " ")
        doc = doc.replace(':', " ")
        lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in doc.split(" ")])
        #stemmered_output = " ".join([stemmer.stem(w) for w in lemmatized_output.split(" ")])
        #preproccessed_doc.append(" ".join(list(dict.fromkeys(lemmatized_output.split(" ")))))
        #print(lemmatized_output)
        preproccessed_doc.append(lemmatized_output)
    return preproccessed_doc



def bigrammer(doc):
    sentence_stream = simple_preprocess(strip_tags(doc), deacc=True)
    return bigram_phraser[sentence_stream]
counter=1
for shot in list:
    shot=text_cleanner(shot)
    shot=lemmatizering(shot)
    sentence_stream = [doc.split(" ") for doc in shot]
    bigram = Phrases(sentence_stream, min_count=5, threshold=100, delimiter=b' ')
    bigram_phraser = Phraser(bigram)
    model = Top2Vec(shot, tokenizer=bigrammer, speed="deep-learn")
    model.save("Topic-Model-arxiv-deep"+str(counter))
    counter += 1