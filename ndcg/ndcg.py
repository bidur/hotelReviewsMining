# http://pythonexample.com/search/sklearn-metrics/15

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


	
def write2File(dataStr,fileName='0_ndgc_summary.csv', fileMode='a'):
	f=open(fileName,fileMode)
	f.write( dataStr)
	#f.write('\n')
	f.close()


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
 
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
 
    y_score : array-like, shape = [n_samples]
        Predicted scores.
 
    k : int
        Rank.
 
    gains : str
        Whether gains should be "exponential" (default) or "linear".
 
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best
    
    
def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
 
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
 
    y_score : array-like, shape = [n_samples]
        Predicted scores.
 
    k : int
        Rank.
 
    gains : str
        Whether gains should be "exponential" (default) or "linear".
 
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
 
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")
 
    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
 

def printResults(df, reviewCountThreshold = 0):
	
	real = list(df.realRating[df.reviewCount > reviewCountThreshold]) 
	predicted = list(df.aspectScore[df.reviewCount > reviewCountThreshold])
	ndcgValue =  ndcg_score(real, predicted)
	print ('ndcg: %.2f' % ndcgValue)
	spearCoef, p = spearmanr(real, predicted)
	print('Spearmans Coeff: %.3f' % spearCoef) , 
	print('   p value =%.3f' % p)
	csvData = str(reviewCountThreshold)+','+ str( np.around(ndcgValue,decimals=2) ) +','+  str( np.around(spearCoef,decimals=2) )+','+  str( np.around(p,decimals=2) )
	write2File('\n'+ csvData,fileName='0_ndgc_summary.csv', fileMode='a')
	print '___________________________________________'
	
 


def compareResults(inputFile):
	
	write2File('Aspect (normalized) = LDA + Cooccurrence Matrix of Frequent Noun from 3 grams  \n ReviewCount, ndgc, Spearman , Spearman_p-value ',fileName='0_ndgc_summary.csv', fileMode='w')	
	sheet_name='Sheet1'
	df = pd.read_excel(inputFile, sheet_name)
	printResults(df, reviewCountThreshold = 200)
	printResults(df, reviewCountThreshold = 100)
	printResults(df, reviewCountThreshold = 50)
	printResults(df, reviewCountThreshold = 0)
	
	
	
	


compareResults("_2X_3gram_result_reviews0.xlsx")
