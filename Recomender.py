import numpy as np
import pandas as pd
import ast

# Adding Dataset
movies = pd.read_csv('Data\movies.csv')
credits = pd.read_csv('Data\credits.csv')

#Preprocessing

movies = movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


movies.dropna(inplace=True)
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
def convert3(obj):
    L=[]
    counter = 3
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L




movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#for removing spaces between words
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x:" ".join(x))
new['tags'] = new['tags'].apply(lambda x:x.lower())

#print(new.head())



#For creating roots
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new['tags'] = new['tags'].apply(stem)

#print(new.head())

#Vectorization

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()


#finding cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
#print(cosine_similarity(vectors).shape)

#recomender
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

#dumping stuff
import pickle
pickle.dump(new,open('Data\movie_list.pkl','wb'))
pickle.dump(similarity,open('Data\similarity.pkl','wb'))