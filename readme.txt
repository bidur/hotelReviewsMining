Mining hotel reviews( from tripAdvisor and Yelp) to identify latent hotel aspects using algorithms like LDA, word2vec, n-gram, etc.
Latent hotel aspects identification using differnet algorithms. The hotel reviews from TripAdvisors and Yelp are taken as input.
0. data ( the OpinRank_TA_review.txt.gz is minimized as github cannot maintain file larger than 50 MB)
1. LDA: for topic modeling using opinRank hotel review dataset
2. word2vec: clustering opinRank hotel reviews to identify clusters using word2vec and kmeans algorithm
3. estimateHotelRank:
		3.1 Baseline method: count based method using basic implementation of LDA and word3vec ( base-LDAw2v_CountBased.py )
		3.2 Sentiment based score: scoring mehtod based on textblob sentiment score ( aspect-LDAw2v_sentimentScore_Test.py )
		3.2 Cosine similarity based score: scoring method based on cosine similarity of the hotel review with positive wordlist and negative word list ( aspect-LDAw2v_COSSimScore_Test.py )
4. ndcg: ndcg applied to the resut of estimateHotelRank
5. regression: Regression applied to the result given by estimateHotelRank ( e.g. _2X_3gram_ALL_HOTEL_TOPIC_VALUES_0_r.csv) to estimate the aspect weights ( i.e. regression coefficients).
	Once the weights are estiated, then the final calculation is done using those coefficents in (3. estimateHotelRank)
