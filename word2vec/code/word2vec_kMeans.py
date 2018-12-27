# -*- coding: utf-8 -*-
 
from gensim.models import Word2Vec
 
from nltk.cluster import KMeansClusterer
import nltk
import os
import gensim
import gzip
import pandas as pd
from sklearn import cluster
from sklearn import metrics

import re
from gensim import  corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')

################  Adjustible parameters ############### 

NUM_CLUSTERS=20
WORDS_PER_TOPIC = 20
################  Adjustible parameters ############### 


def clean_text(tokenized_text, filterPOS=''): #'noun','adjective'
   
    #tokenized_text = word_tokenize(text)
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    
    if(filterPOS == 'noun'):
        tags = nltk.pos_tag(cleaned_text)
        # filter nouns
        cleaned_text = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        # get lemma of the words
        cleaned_text = [lmtzr.lemmatize(word) for word in cleaned_text ]  # return lemma because we want 'rooms' and 'room' as same thing
  
    return cleaned_text


def write2File(dataStr,fileName='data.csv', fileMode='a'):
	f=open(fileName,fileMode)
	if dataStr != '':
		f.write( dataStr)
		#f.write('\n')
	f.close()
	
def readInput( inputFile):
    data = []
    
    documents = list(readData(inputFile))
    #document = ' '.join(readData(inputFile))
    #data.append(document)
    return documents

def readData(textFile):
    # Directly Read gzip files ( as less storage requirement than plain text files)
    print("Reading :", textFile)
    with gzip.open(textFile, 'rb') as f:
        for i, line in enumerate(f):           
            # do some pre-processing and return list of words for each review
            yield gensim.utils.simple_preprocess(line.lower())
    
    print("Reading Input Complete")

# training data   
inputDataFile ='OpinRank_TA_review.txt.gz'

abspath = os.path.dirname(os.path.abspath(__file__))
inputFile = os.path.join(abspath, "../../data/"+inputDataFile)
data = readInput( inputFile)
# For gensim we need to tokenize the data and filter out stopwords

tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text, filterPOS='noun'))
print tokenized_data
'''
tokenized_data = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
            ['this', 'is',  'another', 'book'],
            ['this', 'is', 'hotel'],
            ['hotel', 'staff', 'rude', 'to', 'guests'],
            ['good', 'staff', 'service'],
            ['this', 'room', 'is', 'new', 'one'],
            ['nasty', 'customer',  'service', 'staff'],
            ['free', 'wifi', 'hotel'],
            ['internet', 'and', 'staff', 'not', 'good'],
            ['worst', 'place', 'stay'],
            ['best', 'place', 'to', 'spend', 'night'],
            
            ['this', 'breakfast', 'has', 'tea', 'coffee', 'bread'],
            ['nice', 'room', 'stay'],
            ['hotel', 'room', 'with', 'good', 'breakfast'],
            ['open', 'space', 'big', 'room', 'for', 'stay'],  
            ['hotel', 'booked', 'from', 'the', 'home', 'me']]

'''

# training model
# gensim word2vec - default parameters given here:   
#  https://github.com/danielfrg/word2vec/blob/master/word2vec/scripts_interface.py
# https://www.quora.com/Whats-the-best-word2vec-implementation-for-generating-Word-Vectors-Word-Embeddings-of-a-2Gig-corpus-with-2-billion-words
# 
#default model : CBOW (predicts a words given its context), iteration: 5, size:100 , window:5 , alpha:Set the starting learning rate; default is 0.025
model = Word2Vec(tokenized_data, min_count=2) # min_count=Minimium frequency count of words, 
 
# get vector data
X = model[model.wv.vocab]  #model.wv.vocab may be deprecated , if problems then check model.vocab instead?

 
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print '-----------------------------------------------' 
print ("Cluster id labels for inputted data")
print (labels)
#print ("Centroids data")
#print (centroids)
outputFile = 'w2v_'+ inputDataFile+'_Topics-'+ str(NUM_CLUSTERS ) + '_Nouns-'+ str(WORDS_PER_TOPIC) +'.txt'
write2File('',fileName=outputFile, fileMode='w')

topicDF = pd.DataFrame(columns=['topic', 'word', 'probability'])
topicCtr = 0

for ctr in range(len(centroids)):
	
    printLine = 'centroid: '+ str(ctr)
    currentTopic = model.similar_by_vector(centroids[ctr], topn= WORDS_PER_TOPIC)
    for eachWord in currentTopic:
        topicDF.loc[topicCtr] = [ ctr , eachWord[0] , eachWord[1] ]
        topicCtr += 1
	 
    printLine += str( currentTopic )
    #printLine += str( model.similar_by_vector(centroids[ctr], topn = WORDS_PER_TOPIC) )
    write2File(printLine +'\n',fileName=outputFile, fileMode='a')
    print printLine
    # item = model.similar_by_vector(centroids[0], topn=2)
    # item[0] -> (u'floor', 0.9999650120735168)
    # item[0][0] -> u'floor'

topicDF.to_excel(outputFile+'.xlsx') 

print '-----------------------------------------------' 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
 
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)

#http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
