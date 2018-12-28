input folder: ../data/

Input parameters:

# opinion lexicon
inputFileNeg = 'data/opinion-lexicon-English/negative-words.txt'
inputFilePos = 'data/opinion-lexicon-English/positive-words.txt'

# input hotel info
inputFile = '../data/hotelInfo_TA.xlsx' # hotelInfo_YELP100.xlsx

# input hotel reviews
inputFile = '../data/TripAdvisor100.xls' #yelp_Review100_OLD.xlsx
sheet_name='TA'	#yelp_review100

# Aspect weight( i.e. regression coefficients and intercept)

INTERCEPT= ?
COEFFICIENTS = [..??..]

# other parameters

	aspectWeight = 'same' # wt for normalized/not normalized
	normalizeFlag = 'YES'  # YES/NO . if YES -> totalAspectScore is deivide by totalReviews 
	topicMethod = 'w2v' #'LDA' or 'w2v'
	scoreMethod = 'cos' # cosine similarity or sentiment score
	

	
	
	minimumReviewCount = 0 # Aspect and Aspect weight estimation only for hotels with minimum number of reviews
		
	nGramNumber = 3
	minimumWordRepetition = 2 # n-grams with less than minimumWordRepetition will be ignored
	
	##### READ LDA TOPICS
	nounTopicFileLDA = '../data/LDA_OpinRank_topics-20_NOUN-20_pass-20.txt'
	nounTopicFilew2v = '../data/OpinRank_TA_review.txt.gz_w2v_kMeans_Topics-20_Nouns-20.txt.xlsx'
	probThreshold = 0.01 # minimum probability of Word in Topic
