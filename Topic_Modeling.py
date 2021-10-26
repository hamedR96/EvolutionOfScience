import pandas as pd
from nltk.stem import WordNetLemmatizer
from top2vec import Top2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')
nltk.download('wordnet')

def read_file(csv_file):
    df=pd.read_csv(csv_file,sep=';',low_memory=False)
    df.dropna(subset = ["item6"], inplace=True)
    return df['item6'].tolist()

#PREPROCESSING
def text_cleanner(list):
    text_clean = [re.sub(r'http\S+', '', t) for t in list]
    text_clean= [t.strip().replace('\n', ' ') for t in text_clean]
    docs = [re.sub(r'[\w\.-]+@[\w\.-]+', '', t) for t in text_clean]
    docs = [t.strip().replace('`', ' ') for t in docs]
    return [t.strip().replace('--', ' ') for t in docs]

# JOINT STEMMING & LEMMATIZING
def lemmatizering(list):
    preproccessed_doc=[]
    for doc in list:
        doc=doc.replace('.', " ")
        doc = doc.replace(',', " ")
        doc = doc.replace(';', " ")
        doc = doc.replace(':', " ")
        lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in doc.split(" ")])
        preproccessed_doc.append(lemmatized_output)
    return preproccessed_doc

# Code for Bigram words
def bigrammer(doc):
    sentence_stream = simple_preprocess(strip_tags(doc), deacc=True)
    return bigram_phraser[sentence_stream]


list_of_documents=read_file('arxiv_CS_1998_2017.csv')
doc_list_clean =text_cleanner(list_of_documents)
lemmatizer = WordNetLemmatizer()
preproccessed_doc=lemmatizering(doc_list_clean)
sentence_stream = [doc.split(" ") for doc in  preproccessed_doc]
bigram = Phrases(sentence_stream, min_count=5, threshold=100, delimiter=b' ')
bigram_phraser = Phraser(bigram)

#Training the model and save it
model = Top2Vec(preproccessed_doc,tokenizer=bigrammer,speed="deep-learn")
model.save("Topic-Model-arxiv-deep")

#Loading the model
#model = Top2Vec.load("trained_data")

print(model.get_num_topics())