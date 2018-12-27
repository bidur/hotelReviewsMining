# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import gzip
import os
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

def readData(textFile):
    # Directly Read gzip files ( as less storage requirement than plain text files)
    print("Reading :", textFile)
    with gzip.open(textFile, 'rb') as f:
        for i, line in enumerate(f): 
			#print line 
			# Unicode Encoding Problem: strip out (ignore) the characters giving problem returning the string without them  
			line = unicode(line, errors='ignore')  # unicode(raw, errors='ignore') have the same effect  
			yield (line)
    
    print("Reading Input Complete")

    
def readInput( inputFile):
    data = []
    
    documents = list(readData(inputFile))
    #document = ' '.join(readData(inputFile))
    #data.append(document)
    return documents


def clean_text(text, filterPOS=''): #'noun','adjective'
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    
    if(filterPOS == 'noun'):
        tags = nltk.pos_tag(cleaned_text)
        # filter nouns
        cleaned_text = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] #https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
        # get lemma of the words
        cleaned_text = [lmtzr.lemmatize(word) for word in cleaned_text ]  # return lemma because we want 'rooms' and 'room' as same thing
  
    return cleaned_text

################  Adjustible parameters ############### 
NUM_TOPICS = 20
WORDS_PER_TOPIC = 20
LDA_ITERATIONS = 1 #20
################  Adjustible parameters ############### 


STOPWORDS = stopwords.words('english')

abspath = os.path.dirname(os.path.abspath(__file__))
inputFile = os.path.join(abspath, "../data/OpinRank_TA_review.txt.gz") # set path to inout file
#inputFile = os.path.join(abspath, "review_yelp_TA72.txt.gz")

data = readInput( inputFile)
 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for text in data:
    tokenized_data.append(clean_text(text, filterPOS='noun'))
 
 
# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)
 
# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]


# Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes= LDA_ITERATIONS) #passes= 1 by default? -> MORE PASSES WILL INCREASE ACCURACY

# Build the LSI model
#lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")
 
for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, WORDS_PER_TOPIC))
 
print("=" * 20)

### these printed topics are saved as  identified topics

## TESTING 

from gensim.test.utils import datapath
temp_file = datapath("lda_model.OpinRank_TA_review_passes20")
#temp_file = C:\Python27\lib\site-packages\gensim\test\test_data\lda_model
lda_model.save(temp_file) # C:\Python27\lib\site-packages\gensim\test\test_data\lda_model
#  load model as 'lda'
#lda = models.LdaModel.load(temp_file)


#  CHECK models to work and transform unseen documents to their topic distribution:
text = "The hotel was nice but it is located far from the station"
bow = dictionary.doc2bow(clean_text(text))
print(lda_model[bow])

#performing similarity queries using topic models

from gensim import similarities
lda_index = similarities.MatrixSimilarity(lda_model[corpus])
# Make some queries
similarities = lda_index[lda_model[bow]]
# Sort the similarities
similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
# Find the Top most similar documents:
print(similarities[:10])
# Which is the most similar document
document_id, similarity = similarities[0]
print(data[document_id][:30000])


# optimize and validate coherence lda : https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
