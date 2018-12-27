# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
import nltk
import collections
#from nltk.tokenize import MWETokenizer

import string
import os
import gc
import numpy as np
from collections import Counter
from textblob import TextBlob  # float(TextBlob('bad').sentiment.polarity) # polarity range [-1,1]

from math import log
from scipy.stats import spearmanr





inputFileNeg = '../data/opinion-lexicon-English/negative-words.txt'
inputFilePos = '../data/opinion-lexicon-English/positive-words.txt'
	


def readWordAsList(inputPosNegFile):
    f = open(inputPosNegFile)
    words = [ line.rstrip() for line in f.readlines()]
    
    return list(set(words))
    
posList = readWordAsList(inputFilePos)
negList = readWordAsList(inputFileNeg)


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def removeNonAlphaNumericChar(textStr):
	s = re.sub('[^0-9a-zA-Z]+', '', str(textStr)) # remove all characters except alpha-numeric
	return ''+s


def cleanLine(line):
		
	return line.replace("\n","").replace("\r","").replace("\\","").replace(")","").replace(" ","").replace('"',"").replace("'","")

def cleanNumber(text):
	return float(re.sub(r'[^0-9\.]', '', text)) 
	
# Removes all the non ASCII characters, return only ASCII characters
def removeNonASCII(string):
    cleaned = (c for c in string if 0 < ord(c) < 127)
    return ''.join(cleaned)	
	 







# check if noun and adj present in the input, if present generates a list of nouns and adj and returns
def check_noun_adj_phrase(n_gramList, n=2):
	
	adjList =[]
	nounList=[]
	lastWordTag ='' # if NN is not at last , then accept it else ignore
	for i in n_gramList:
		i_tag = nltk.pos_tag([i])
		if (i_tag[0][1] == 'NN'):
			nounList.append(i)
			
		elif(i_tag[0][1] == 'JJ'): 
			adjList.append(i)
			
		lastWordTag = i_tag[0][1]
		
	#if( len(nounList) + len(adjList) == n) and lastWordTag=='NN': # Last word must be NN and n gram contains NN and ADJECTIVES only and not other words
	if( len(nounList) + len(adjList) == n) : # n gram contains NOUN and ADJECTIVES only and not other words
		return adjList,nounList
	else:
		return [],[] # empty list returned as at least 1 word is not NOUN/ADJECTIVE 




# separate noun and adj from input  and return a list of nouns and adj 
def get_noun_adj_list(wordList):
	
	adjList =[]
	nounList=[]
	for i in wordList:
		i_tag = nltk.pos_tag([i])
		if (i_tag[0][1] == 'NN'):
			nounList.append(i)
			
		elif(i_tag[0][1] == 'JJ'): 
			adjList.append(i)
			
	return nounList, adjList
	
	
#  dont replace fullstop and comma	from input
def handle_punctuationc(s):
    #punc_list = ["-",";",":","!","?","/","\\","#","@","$","&",")","(","'","\""] # dont replace fullstop and comma
    punc_list = string.punctuation.replace('.','').replace(',','')

    new_s = ''
    for i in s:
        if i not in punc_list:
            new_s += i
        else:
            new_s += ' '
    return new_s.lower()



# nltk.sent_tokenize() cannot handle cases with dots like: hotel.room, M.r. etc.
# Use split_into_sentences() instead of  nltk.sent_tokenize() 
#https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
def split_into_sentences(text):
	caps = "([A-Z])"
	prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
	suffixes = "(Inc|Ltd|Jr|Sr|Co)"
	starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
	acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
	websites = "[.](com|net|org|io|gov)"
	
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences

# write coocurrance matrix to screen and csv file
def generateMatrixNNJJ( nGramList, nounList, adjList, n, outputFile):
	
	# Make NN-JJ co-occurance matrix
	cMat = [0] * len(nounList) * len(adjList) # noun-adj co-occurance matrix
	cMat = np.reshape (cMat,( len(nounList),len(adjList) ) )
	for wordList,count in nGramList.items():
		nl,al = get_noun_adj_list(wordList)# list of nouns and adj 		
		for noun in nl:
			for adj in al:
				cMat [nounList.index(noun)][adjList.index(adj)] += count # update the count 
			

	
	
	
	# print Matrix , write to csv
	#matrixFile = open(outputFile +"_MATRIX.csv","w")
	printLine = "\n Nouns: "+ str( len(nounList)) + " , " + str(nounList) 
	#matrixFile.write(printLine+'\n' )
	#print printLine
	
	
	opLine = " " + ','
	#print(' ' * 17),
	for adj in adjList:
		opLine += adj +','
		#print('{:7}'.format(adj)),
		
	#matrixFile.write(opLine +"\n")
	opLine=''
	#print '\n'
	for j in range(0, len(nounList)):
		for i in range( 0, len(adjList)):
			if i==0:
				#print '{:10}'.format(nounList[j]) +':', 
				opLine = nounList[j] +','
			#print '{:7}'.format(cMat[j][i]) ,
			opLine += str( cMat[j][i] ) +','
		#print '\n'
		#matrixFile.write(opLine +"\n")
		opLine=''
	
	#matrixFile.write(opLine +"\n")
	#matrixFile.close()
	
	del(nGramList, nounList, adjList)
	gc.collect()
	return cMat # return the matrix
	
###################################################################	





def get_ngram(raw , n , minimumWordRepetition): # raw : list of Reviews. each Reviews is multi sentences
	nounSet = set()
	adjSet = set()
	nGramFinal ={}
	stopwords = set(nltk.corpus.stopwords.words('english') ) # list are slower than set , so convert to set
	
	
	# replace punctuation by space
	raw = handle_punctuationc(raw) ####
	
	
	# Unicode Encoding Problem: strip out (ignore) the characters giving problem returning the string without them
	raw = unicode(raw, errors='remove')  ##### unicode(raw, errors='ignore') have the same effect	
		
	sentences_N_list = split_into_sentences(raw)# each sentence in rawdata  is separated . The nltk.sent_tokenize() have some flaws
	#print sentences_N_list
	
	#opFile = open(outputFile +".csv","w")
	nGramList = []
	#print len(sentences_N_list)
	
	for eachSentence in sentences_N_list:
		
		eachSentence = eachSentence.lower() 
		
		tokens = nltk.word_tokenize(eachSentence) # word_tokenize() cannot handle compound words like 'wi fi', 
		

		#filter stopwords
		tokens = [word for word in tokens if word not in stopwords]
		
	
		#b_grams = nltk.bigrams(tokens)
		nGms = nltk.ngrams(tokens, n)
		for each_nGms in nGms:
			nGramList.append(each_nGms)
		
		
	#compute frequency distribution for all the n-grams in the text	
	fdist = nltk.FreqDist(nGramList)
	#print Counter(fdist)
	#print '-------------------------------------------------'
	for wordList,count in fdist.items():
		#print count
		if count >= minimumWordRepetition:# ignore ngram with less than minimumWordRepetition occurance
			adjList, nounList = check_noun_adj_phrase(wordList, n)
			if(len(adjList) >0 and len(nounList)>0 ):	# noun and adj present in the n-gram
				printLine = str(wordList )+ "," + str(count) # printLine contains the desired n-gram 
				#opFile.write(printLine +"\n")
				#print printLine
				nounSet = nounSet.union(set(nounList))
				adjSet = adjSet.union(set(adjList))
				nGramFinal[wordList] =  count
			#print '.',
						
	#opFile.close()
	del(fdist)
	gc.collect()
	return nGramFinal, sorted(nounSet), sorted(adjSet)	# sorted function to a set returns a sorted list
	


def createOutputDir(path):
	
	try:  
		if os.path.exists(path):
			print( " Dir exists: "+ path)
			return
		else:
			os.mkdir(path)
			print( " %s created" % path)
		
	except OSError:  
		print ("Creation of the directory %s failed" % path)
	
def   getOverallHotelRating():
	hotelRatingDict={}
	hotelReviewCountDict= {}
	inputFile = '../data/hotelInfo_TA.xlsx' # hotelInfo_YELP100.xlsx
	sheet_name='hotelInfo'
	
	df = pd.read_excel(inputFile	, sheet_name)
	hotelNames = df['name']
	hotelRatings = df['hotel_rating']
	hotelReviewCount = df['review_count']
	for i in range(len(hotelNames)):
		hotelRatingDict[hotelNames[i] ] = hotelRatings[i]
		hotelReviewCountDict [hotelNames[i] ] = hotelReviewCount[i]
		
	return hotelRatingDict, hotelReviewCountDict


# get aspects and adjectives from LDA Topics
def readLDATopicFile(inputFile, outputDir, probThreshold=0):
	
	allLDATopicsFile = outputDir + '0_topics_LDA_'+ inputFile.replace('/','').replace('.txt','')
	#write2File("LDA TOPICS: \n"+ '\n',fileName = allLDATopicsFile , fileMode='w')
	f = open(inputFile, "r")
	textLines = list(f)
	f.close()
	
	aspectDictByTopic = {}  # aspects and adjectives from LDA topics  , aspectDictByTopic[0] gives list of aspects probabilities in Topic#0
	aspectPDictByTopic = {}
	aspectCtr = 0
	for i in range(len(textLines)):
		#print i
		if len(textLines[i].strip()) == 0 :# if line is empty 
			continue
			
		temp = textLines[i].split(',') # [
		temp2 =  temp[1].split(' + ') # array of values: 0.032*"center" # ), (
		currrentTopicList = []
		currrentProbList = []
		for item in temp2: 
			item = cleanLine(item)
			item2 =  item.split('*') # ,
			## filter by prob 
			if (cleanNumber(item2[0]) > probThreshold): 
				currrentTopicList.append(cleanLine(item2[1]))
				currrentProbList.append( cleanNumber(item2[0]) )
		
		aspectDictByTopic[aspectCtr] = currrentTopicList
		aspectPDictByTopic[aspectCtr] = currrentProbList
		
		#write2File('topic_'+ str(aspectCtr) +' = '+ str(currrentTopicList) + '\n',fileName = allLDATopicsFile , fileMode='a')
		
		aspectCtr += 1
	
	aspectDictByTopic2 = refineLDATopicsByWordProb ( aspectDictByTopic, aspectPDictByTopic)  # remove duplicates, keep word with max prob only
	
	pd.DataFrame.from_dict(aspectDictByTopic2, orient='index').to_csv(allLDATopicsFile+'.csv')
	
	return (aspectDictByTopic2)



def findDuplicateWordInDict(aspectDictByTopic):
	
	gAspectList = [ aspectDictByTopic[i] for i in range(0,len(aspectDictByTopic)) ]   # 2-d array of Aspects
	globalAspectList = [j for sub in gAspectList for j in sub] # flattern to 1-D list

	duplicateWords =  [item for item, count in collections.Counter(globalAspectList).items() if count > 1]
	return duplicateWords

def refineLDATopicsByWordProb(aspectDictByTopic, aspectPDictByTopic): # remove duplicates, keep word with max prob only
	# identify duplicates

	duplicateWords =  findDuplicateWordInDict(aspectDictByTopic)

	
	duplicateWordLocDict = {} # duplicateWordLocDict['bed'] =6 -> keep bed on topic 6 only , remove bed from other topics


	# get location of duplicates

	maxWordProb = {}

	maxWordProb = {item: -1.0 for item in duplicateWords } # initialize
	duplicateWordLocDict   = {item: -1.0 for item in duplicateWords } # initialize

	for item in duplicateWords:
		for i in range( len(  aspectDictByTopic.keys() ) ):
			currentTopic = aspectDictByTopic[i]
			if item in currentTopic:
				#print item,
				#print i,
				itemIndex =  currentTopic.index(item)
				#print itemIndex,
				
				#print aspectPDictByTopic[i][itemIndex] # prob
				
				
				if maxWordProb[item] < aspectPDictByTopic[i][itemIndex]:
					maxWordProb[item] = aspectPDictByTopic[i][itemIndex]
					duplicateWordLocDict[item] = i ##################################
					
					
					
	# remove duplicate word, but keep a single copy with the MAX probability
	
	for i in range( len(  aspectDictByTopic.keys() ) ):
		currentTopic = aspectDictByTopic[i]		
		for item in duplicateWordLocDict.keys():
			if duplicateWordLocDict[item] != i:  # if duplicateWordLocDict[item] == i ->  keep it as this is the maximum word probability case
				if item in aspectDictByTopic[i]:
					itemIndex =  currentTopic.index(item)
					del aspectDictByTopic[i][itemIndex]
					
				

	return aspectDictByTopic






# get aspects and adjectives from w2v Topics
def readw2vTopicFile(inputFile, outputDir, probThreshold=0):
	
	allLDATopicsFile = outputDir + '0_topics_w2v'+ inputFile.replace('/','')
	write2File("w2v TOPICS: \n"+ '\n',fileName = allLDATopicsFile , fileMode='w')
	
	sheet_name='Sheet1'
	df = pd.read_excel(inputFile, sheet_name)
	topicList = list(df['topic'])
	wordList = list(df['word'])
	probList = list(df['probability'])
	
	aspectDictByTopic = {}  # aspects and adjectives from LDA topics  , aspectDictByTopic[0] gives list of aspects probabilities in Topic#0
	totalTopics = len(set(topicList))
	for i in range (totalTopics):
		aspectDictByTopic[i] = df.word[ (df['topic'] == i ) & (df['probability'] > probThreshold) ]
		write2File('topic_'+ str(i) +' = '+ str(aspectDictByTopic[i]) + '\n',fileName = allLDATopicsFile , fileMode='a')
			
	#print aspectDictByTopic
	return (aspectDictByTopic)
	



def write2File(dataStr,fileName='data.csv', fileMode='a'):
	f=open(fileName,fileMode)
	f.write( dataStr)
	#f.write('\n')
	f.close()




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calcCosineSimilarity(strData):
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(strData)
	result_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
	
	return 0 if result_cos is None else result_cos[0][1]
	

######################################################
'''
AspectScoreNormalize = 1 ( Default: no noramalization of the AspectScore by number of reviews)
AspectScoreNormalize = number of reviews (Bias in Score for different hotel due to unequal number of reviews is neutralized)
'''
def estimateHotelRating_method_CosSim(currentHotelReviewsAsStr, currentHotel,  aspectDictByTopic,  hotel_NNJJ_Mat, nounList4Hotel, adjList4Hotel, AspectScoreNormalize=1):
	
	
	
	gAspectList = [ aspectDictByTopic[i] for i in range(0,len(aspectDictByTopic)) ]   # 2-d array of Aspects
	globalAspectList = [j for sub in gAspectList for j in sub] # flattern to 1-D list
	aspectsPresent = list( set(globalAspectList) & set(nounList4Hotel) ) # aspects Present for this hotel
	
	aspectValuesDict = {}
	#initialize with aspects/nouns of the current hotel 
	aspectValuesDict = {key: INTERCEPT for key in aspectsPresent } # all items are set to 0 e.g: aspectValuesDict['room'] = 0
	
	currentHotelAspectNames = []
	######################################
	reviewSentList = split_into_sentences( currentHotelReviewsAsStr )
	for eachLine in reviewSentList:
		posData = (eachLine, str(posList))
		posSim = calcCosineSimilarity(posData)
		
		negData = (eachLine, str(negList))
		negSim = calcCosineSimilarity(negData)
	
		sentenceScore =  posSim -  negSim  ###################
		
		wordCounter = Counter(eachLine.split())
		
		for word, wordFreq in wordCounter.iteritems():
			word = word.rstrip('.,?!\n') # removing possible punctuation signs
			
			if word in aspectsPresent:
				aspectWeight = getAspectWeight4word(aspectDictByTopic, word)
				aspectValuesDict [ word ] += aspectWeight * wordFreq * sentenceScore  #### Apply aspect weight here
				currentHotelAspectNames.append(word)
				
		
	###################################################################
	
	aspectValuesDictSorted = collections.OrderedDict(sorted(aspectValuesDict.items()))
	for key in aspectValuesDictSorted.keys():
		aspectValuesDictSorted[key]=aspectValuesDictSorted[key]/AspectScoreNormalize
		
	print '                         estimateHotelRating_method_CosSim: ' +currentHotel
	
	return aspectValuesDictSorted, str(list(set(currentHotelAspectNames)) ) # aspect and respective weights in a dict
	
					
		
def processEachHotel(reviewsDF, minimumReviewCount,nGramNumber, minimumWordRepetition, aspectDictByTopic, opFileName, AspectScoreNormalizeFlag = 'YES' ):
	#General Hotel Informations
	hotelRatingDict , hotelReviewCountDict = getOverallHotelRating()
	
	allReviewList = reviewsDF['review_text']
	allTitleList = reviewsDF['review_title']
	allHotelsDataAsStr = ". ".join(map(str,allReviewList)) + ". ".join(map(str,allTitleList)) # optimize by sending the list here and handling the list in get_ngram()
	hotelNameList = sorted(set(reviewsDF['name']))
	
		
	allHotelDetailFile = opFileName+'ALL_HOTEL_TOPIC_VALUES_'+str(minimumReviewCount)+'_r.csv'
	
	fileHeader = 'hotel,Rating,totalScore'
	for i in range(20):
		fileHeader += ",Asp_"+str(i) 
	fileHeader += "\n" 
	
		
	write2File(fileHeader,fileName = allHotelDetailFile, fileMode='w')
	
	allHotelAspectFile = opFileName+'ALL_HOTEL_TOPIC_VALUES_'+str(minimumReviewCount)+'.csv'
	write2File(fileHeader,fileName = allHotelAspectFile, fileMode='w')
	
	
	resultDF = pd.DataFrame(columns=['hotel', 'realRating', 'aspectScore','reviewCount'])
	resultRow = 0
	
	
	
	for currentHotel in hotelNameList:
		
		
		
		if (currentHotel in hotelReviewCountDict.keys() and hotelReviewCountDict[currentHotel] < minimumReviewCount): # if the hotel is not in hotelINfo file then skip
			continue
			
		currentHotelDataAsStr=''
		
		opFileNameHotel = opFileName+removeNonAlphaNumericChar(currentHotel)
		
		allReviewList4Hotel = reviewsDF.review_text[reviewsDF.name == currentHotel] 
		allTitleList4Hotel = reviewsDF.review_title[reviewsDF.name == currentHotel] 
		currentHotelDataAsStr = ". ".join(map(str,allReviewList4Hotel)) + ". ".join(map(str,allTitleList4Hotel)) # optimize by sending the list here and handling the list in get_ngram()
	
		nGramListGlobal, nounList4Hotel, adjList4Hotel = get_ngram( currentHotelDataAsStr, nGramNumber, minimumWordRepetition) # nGramNumber=3 -> 3gram
		hotel_NNJJ_Mat= generateMatrixNNJJ( nGramListGlobal, nounList4Hotel, adjList4Hotel, nGramNumber, opFileNameHotel) # generate and print/write cooccurance matrix to csv
		
		#estimateHotelRating_method_Sentiment ( currentHotel,  aspectDictByTopic,  hotel_NNJJ_Mat, nounList4Hotel, adjList4Hotel , AspectScoreNormalize = hotelReviewCountDict[currentHotel])
		
		# AspectScoreNormalize by hotelReviewCount ( Default is not normalize when AspectScoreNormalize = 1)		
		if AspectScoreNormalizeFlag == 'YES':			
			
			aspectValuesDict, currentHotelAspectNames = estimateHotelRating_method_CosSim(currentHotelDataAsStr, currentHotel,  aspectDictByTopic,  hotel_NNJJ_Mat, nounList4Hotel, adjList4Hotel , AspectScoreNormalize = hotelReviewCountDict[currentHotel])
		else:
			
			aspectValuesDict, currentHotelAspectNames = estimateHotelRating_method_CosSim( currentHotelDataAsStr, currentHotel, aspectDictByTopic,  hotel_NNJJ_Mat, nounList4Hotel, adjList4Hotel, AspectScoreNormalize = 1  )
		
		eachTopicValue4aHotel =  sum(aspectValuesDict.values())
		
		#hotelRating=''
		#hotelReviewCount = ''
		#if currentHotel  in hotelRatingDict.keys():
		hotelRating = str(hotelRatingDict[currentHotel])
		hotelReviewCount = str(hotelReviewCountDict[currentHotel])
		
		writeHotelDataByTopicValue(currentHotel,hotelRating, aspectDictByTopic, aspectValuesDict, allHotelAspectFile, allHotelDetailFile)
		# write to result dataframe
		resultDF.loc[resultRow] = [currentHotel , hotelRating , eachTopicValue4aHotel , hotelReviewCount ]
		resultRow += 1
		print "FINISH: "+currentHotel
		
	resultDF = resultDF.sort_values('realRating',ascending= False)
	resultDF.to_excel(opFileName+'result_reviews'+str(minimumReviewCount)+'.xlsx') 
	print'FINISH ALL HOTELS'
	
	del(reviewsDF, minimumReviewCount,nGramNumber, minimumWordRepetition, aspectDictByTopic, allReviewList, allTitleList, allHotelsDataAsStr, hotelNameList, allReviewList4Hotel, allTitleList4Hotel)
	gc.collect()
	
	return resultDF
	
	
def writeHotelDataByTopicValue(currentHotel,hotelRating, aspectDictByTopic, aspectValuesDict, allHotelAspectFile, allHotelDetailFile): # each Topic = each aspect
	
	alltopicTotal =  sum(aspectValuesDict.values())
	#print currentHotel+'  '+str(alltopicTotal) +  ': ',
	#print aspectValuesDict.values()
	 
	
	printLine  = currentHotel + ','+ hotelRating +' , '+ str( alltopicTotal ) +'\n'
	# write aspects by topic for currentHotel
	for k in range ( len( aspectDictByTopic) ):
		topicSum = 0
		printLine += ',,,'+ 'Topic#'+ str(k) 
		printLine4word = ''
		for aspect in aspectDictByTopic[k]:				
			if( aspect in aspectValuesDict.keys() and aspectValuesDict[aspect] !=0):
				topicSum += aspectValuesDict[aspect]
				printLine4word+=',,,,,'+ str(aspect) +','+ str(aspectValuesDict[aspect]) + '\n'			
		printLine += ',' + str(topicSum) +'\n'	+ printLine4word
	write2File(printLine,fileName = allHotelAspectFile , fileMode='a')
	
	
	printLine2  = currentHotel + ','+ hotelRating +' , '+ str( alltopicTotal ) 
	# write aspects by topic for currentHotel
	for k in range ( len( aspectDictByTopic) ):
		topicSum = 0
		
		for aspect in aspectDictByTopic[k]:				
			if( aspect in aspectValuesDict.keys() and aspectValuesDict[aspect] != 0):
				topicSum += aspectValuesDict[aspect]
				#printLine4word+=','+ str(aspect) +','+ str(aspectValuesDict[aspect]) + '\n'			
		printLine2 += ',' + str(topicSum) 
	
	write2File(printLine2 +'\n',fileName = allHotelDetailFile , fileMode='a')
	
	del(currentHotel,hotelRating, aspectDictByTopic, aspectValuesDict, allHotelAspectFile, allHotelDetailFile)
	gc.collect()
	

# Each topic is an Aspect. so each Topic have a weight (i.e. value in COEFFICIENTS list). 
def getAspectWeight4word(aspectDictByTopic, word):# gets the weight of the Aspect 
	
	for i in range(0,len(aspectDictByTopic)):
		
		if word in aspectDictByTopic[i]:
			return COEFFICIENTS[i]  # assumes that a word is present only 1 time in the aspectDictByTopic
			
		else: 
			return 0
	
#################################################################


# TA:read the input file in pandas
inputFile = '../data/TripAdvisor100.xls' #yelp_Review100_OLD.xlsx
sheet_name='TA'	#yelp_review100
reviewsDF = pd.read_excel(inputFile, sheet_name)

	

INTERCEPT_4w2v_NORM = 0
COEFFICIENTS_4w2v_NORM  = []


INTERCEPT_4w2v_NOT_NORM  = 0		
COEFFICIENTS_4w2v_NOT_NORM = []

'''
### LDA-cos-normalized ( 3 gram case)
	
INTERCEPT = 0.0009
COEFFICIENTS = [ 0.3 , -0.  ,  1.31,  1.77,  0.83,  1.12,  0.89, -0.64, -4.41,
        1.16,  0.29,  4.01,  0.91,  0.25,  0.91,  2.57,  0.97,  0.65,   1.34,  0.71]
        
'''
### w2v-cos- normalized ( 3 gram case)
   
INTERCEPT= 0.0
COEFFICIENTS = [ 0.,  1.,  1.,  1.,  1.,  0.,  1.,  1., -0.,  1., -0., -0., -0.,
        1.,  1., -0.,  1.,  1.,  0.,  1.]

	
def main():
	print "*** START ***"
	
	#aspectWeight = 'same' # wt for normalized/not normalized
	normalizeFlag = 'YES'  # YES/NO . if YES -> totalAspectScore is deivide by totalReviews 
	topicMethod = 'w2v' #'LDA' or 'w2v'
	scoreMethod = 'cos' # cosine similarity
	

	
	
	minimumReviewCount = 0 # Aspect and Aspect weight estimation only for hotels with minimum number of reviews
		
	nGramNumber = 3
	minimumWordRepetition = 2 # n-grams with less than minimumWordRepetition will be ignored
	
	##### READ LDA TOPICS
	#nounTopicFileLDA = 'data/LDA_OpinRank_TA_review_topics-20_words-30_pass-10_noun.txt'
	nounTopicFileLDA = '../data/LDA_OpinRank_topics-20_NOUN-20_pass-20.txt'
	nounTopicFilew2v = '../data/OpinRank_TA_review.txt.gz_w2v_kMeans_Topics-20_Nouns-20.txt.xlsx'
	probThreshold = 0.01 # minimum probability of Word in Topic
	outputDir=''

		
	if normalizeFlag == 'YES':
			
		outputDir = topicMethod+ '_COS_TEST_NORM_minReview-' + str(minimumReviewCount)+ '_wordProb'+'-'+str(probThreshold) +'_'+ str(nGramNumber) +'-gram_'+ str(minimumWordRepetition) +'X/'
	else:
		outputDir = topicMethod+ 'COS_minReview-' + str(minimumReviewCount)+ '_wordProb'+'-'+str(probThreshold) +'_'+ str(nGramNumber) +'-gram_'+ str(minimumWordRepetition) +'X/'
		
			
	createOutputDir(outputDir)# if not exist create
	opFileName = outputDir+'_'+  str(minimumWordRepetition) +'X_'+ str(nGramNumber) +'gram_'	
		
		
		
	if topicMethod == 'LDA':
		aspectDictByTopic = readLDATopicFile(nounTopicFileLDA, outputDir, probThreshold) # aspectDictByTopic[0]=['room','bed'] # aspects of topic#0
		
	else:
		aspectDictByTopic = readw2vTopicFile(nounTopicFilew2v, outputDir, probThreshold)

		
	'''
	AspectScoreNormalizeFlag = YES/NO
	'''
	resultDF = processEachHotel(reviewsDF, minimumReviewCount,nGramNumber, minimumWordRepetition, aspectDictByTopic, opFileName, AspectScoreNormalizeFlag = normalizeFlag)
	# columns=['hotel', 'realRating', 'aspectScore']
	del (aspectDictByTopic, resultDF)
	gc.collect()
	
	print 'Finish for minimumReviewCount: ' ,
	print minimumReviewCount
	
	
	
	
if __name__ == "__main__":
	
	main()

