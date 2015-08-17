import glob
import math
from statistics import *
from sklearn import linear_model
from scipy.sparse import csr_matrix
import numpy as np
import random
#using naive Bayes for Classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import cross_validation
from sklearn import feature_selection
from sklearn.feature_selection import chi2
import argparse
import csv
import sys
csv.field_size_limit(sys.maxsize)
# processing the data output

#from operator import add;
seed=1003
random.seed(seed)

# reads the POS tags of the tweets in path2Data
# Since numbers and such give many unique ''words'' we can specify which to replace
# currently replacing '#'-hashtags, '@'-usernames, 'U'-url links, 'E'-emoticons, '$'- numeral , ','-punctuations, 'G'-unknown tag
# we are removing some formating symbols eg. ":" which is tagged to '~'
replaceables = ['@', 'U', '$', ',', 'G']
removables = ['~']
def cleanTweet(tweet, tweet_pos):
    tweet_l = tweet.split()
    tweet_pos_l = tweet_pos.split()

    if len(tweet_l) != len(tweet_pos_l):
        for i, item in enumerate(tweet_l):
            print (tweet_l[i], ',' , tweet_pos_l[i])
        
    clean_tweet = []
    for i, item in enumerate(tweet_l):
        #print (item)
        #print (tweet_pos_l[i])
        if tweet_pos_l[i] in replaceables:
            clean_tweet.append(tweet_pos_l[i])
        elif tweet_pos_l[i] in removables:
            None
        else:
            clean_tweet.append(item.lower())
            clean_tweet.append(tweet_pos_l[i])
    
    #print (clean_tweet)
    return clean_tweet

# Version 2: generating Valid Keys for corresponding to the top/bottom HIV rates
#sample size is 2 (p*N), p*N for lower and p*N for upper
def sampleItems(locRates, p):
    items = []
    for key in locRates:
        items.append((key, locRates[key]))

    sorted_locRates = sorted(items, key=lambda student: student[1]) 
    total_items = len(items)
    sampleSize = (int) (p*total_items)
    #print ("sample size: ", sampleSize)
    
    ret = []
    lab = {}
    for i, item in enumerate(sorted_locRates):
        if i < sampleSize:
            ret.append(item[0])
            lab[item[1]] = 0
            
        if i >= total_items - sampleSize:
            ret.append(item[0])
            #lab.append(1)
            lab[item[1]] = 1
    
    #print ("samped items size: ", len(ret))
    return (ret,lab)
    

def tfidf(docID, wordID, tf, idf, N):
    tf_0 = 0.5 +  tf[docID].get(wordID, 0)
    idf_0 = math.log( 1 + N/len(idf[wordID]))
    #idf_0 = 1
    return (tf_0 * idf_0)

def conf_mat(Y_hat, Y):
    tp = fp = tn = fn = 0
    for i,j in zip(Y_hat, Y):
        if i == 1:
            if i == j:
                tp = tp + 1
            else:
                fp = fp + 1
        elif i == 0:
            if i == j:
                tn = tn + 1
            else: 
                fn = fn + 1
        else:
            print (" j should only be 0 or 1, however", j , "was encountered.")
    #print (tp, fp, tn, fn)
    return [tp, fp, tn, fn]
    
def getHIVRate(filename, state2rate):
    prefix = filename.split('.')[0]
    prefix_list = prefix.split('/')[-1].split('_')
    #print(prefix_list)
    county = prefix_list[0]
    state = prefix_list[-2]
    my_rate = -1
    if state in state2rate:
        for county_keys in state2rate[state]:
            if county in county_keys:
                #print (county_keys, state2rate[state][county_keys], prefix_list[2])
                my_rate = int(state2rate[state][county_keys])
                
        if my_rate < 0:
            print ('no such county', county, state)
    else:
        print ( 'no such state', state)
    return my_rate

## parameters 
# classification - whether we are using classification or regression
# useIDF - uses the tfidf features (True) otherwise only uses tf
# tf - dictionary for the term frequencies
# idf - dictionary for idf
# N - the total number of locations
def generateX(classification, useIDF, tf, idf, locRates, N,VocabSize):
	x_cord = []

	#sample top/bottom rates
	#sss = [0.25]
	#sss= np.arange(0.05, 0.48, 0.02)
	topPercent = 0.25
	(validKeys, labels) = sampleItems(locRates, topPercent)
	print ("top sample size: ", (int)(topPercent*N), " top: ", topPercent)
	row = []
	col = []
	data = []
	Y = []
	print ("total sample size", len(validKeys))
	for i, docID in enumerate(validKeys):
		for wordID in tf[docID]:
			row.append(i)
			col.append(wordID)
			if( useIDF):
				data.append(tfidf(docID, wordID, tf,idf, N) ) 
			else:
				data.append(tf[docID][wordID])
		# used for classification
		if (classification):
			Y.append (labels[ locRates[docID] ])
		else:
			# to use regression
			Y.append(locRates[docID]) 
    
	X = csr_matrix ( (np.array(data),(np.array(row),np.array(col))), shape=(len(validKeys),VocabSize), dtype=np.dtype('d'))
	print ("shape:", X.shape)
	return (X,Y)

def main():
	parser = argparse.ArgumentParser(description='Script to do experiments on the small dataset.',
		prefix_chars='-+')
	# parser.add_argument('integers', metavar='N', type=int, nargs='+',
	#                    help='an integer for the accumulator')
	parser.add_argument('--classifier', dest='classifier',
	                   default='',
	                   help='choose classifier (Options: NaiveBayes)')

	parser.add_argument('--regression', dest='regression',
	                   default='',
	                   help='choose regressions (Options: ElasticNet)')

	parser.add_argument('--tfidf', action='store_false',
	                   help='turns off the tfidf feature and used tf only')

	parser.add_argument('--pos', action='store_false',
	                   help='adds part of speech tags as features')

	parser.add_argument('--filter', metavar='N', type=int, default=-1,
	                   help='specifies how many features to keep after filtering. Filtering keeps the N number of features')

	parser.add_argument('--minTweets', type=int, default=25,
	                   help='specifies the minimum number of tweets a location should have to consider it in the experiments.')


	parser.add_argument('--path2Data', default='cities/',
	                   help='path to where the tsv files are located. (Default: cities/)')

	args = parser.parse_args()

	if (args.classifier=='' and args.regression==''):
		print (args)
		print("ERROR: not enough parameters, you must specify classification or linear regression.")
		parser.parse_args(['-h'])
		return -1

	if (args.classifier != ''):
		classOrReg = True
	else:
		classOrReg = False;


	i = 0
	heading = []
	state2rate = {}
	for ll in csv.reader( open("hivRatesData/AIDSVu_County_2012.csv", 'r' )):
		if i == 0:
			heading = ll
		else:
			county_rate = int(ll[3])
			if (  county_rate > 0 ): 
				county_name = ll[2]
				state_name = ll[1]
				key = county_name + " " + state_name
				if state_name in state2rate:
					state2rate[state_name][county_name] = str(county_rate)
				else:
					state2rate[state_name] = {county_name : str(county_rate)}
		i = i + 1


	filenum = 0
	Vcount=0
	VVcount=0
	Vinv={}          # index to word map
	V={}             # word to index
	VV={}
	VVinv={}
	idf={}           # forwardIndex
	tf={}            # (word, numberOfWords)
	bitf={}
	locRates={}      # HIV rates based on locations
	locslessthan = 0
	N = 0
	print ("reading files from ", args.path2Data, "...")
	for file in glob.glob(args.path2Data+'*.tsv'):
	    #Getting the HIV rate 
	    my_rate = getHIVRate(file, state2rate)
	    if my_rate < 0:
	        continue
	    
	    lines = []
	    with open(file, 'r') as f:
	        lines = f.readlines()
	    
	    #Checking if this location has less than a threshold number of tweets
	    if len(lines) < args.minTweets:
	        #print(file)
	        locslessthan = locslessthan + 1
	        continue
	        
	    
	    filenum = filenum + 1  #serves as an index for the file name
	    #prefix = file.split('.')[0]
	    #locRates[filenum] = int(prefix.split('_')[-1])
	    locRates[filenum] = my_rate
	    #DEBUG
	    #print (file + " file num: " + str(filenum) + " num tweets: " + str(len(lines)) )

	    unique_words = set([])
	    for line in lines:
	        ll = line.split('\t')
	        tweet = ll[0].strip()
	        tweet_pos = ll[1].strip()

	        prevWord = "<s>"
	        #for word in cleanTweet(tweet, tweet_pos):

	        tweet_features = []
	        if (args.pos):
	        	tweet_features = cleanTweet(tweet,tweet_pos)
	        else:
	        	tweet_features = tweet.split()

	        for word in tweet_features:
	            if word not in V:
	                V[word]= Vcount
	                Vinv[Vcount]=word
	                Vcount = Vcount + 1

	            if V[word] not in idf:
	                idf[ V[word] ] = []

	            if filenum not in tf:
	                tf[filenum] = {}

	            freq = tf[filenum].get(V[word], 0)
	            tf[filenum][ V[word] ] = freq + 1

	            if word not in unique_words:
	                idf[ V[word] ].append(filenum)
	                unique_words.add(word)

	N = filenum
	VocabSize=len(V.keys())

	print ("filtered:", locslessthan, "counties containing less than", args.minTweets, "number of tweets" )
	#some statistics
	print ("vocabSize: ", VocabSize)
	print ("docSize: ", N)
	print ("labels: ", len(locRates))
	mu = mean(locRates.values())
	print ("mean: ",  mu )
	med = median(locRates.values())
	print ("median: ",  median(locRates.values()) )
	print ("max: ", max(locRates.values()) )
	print ("min: ", min(locRates.values()) )
	sigma = stdev(locRates.values(), mu)
	print ("standard diviation: ", sigma )

	#should generate both Y for classification and regression?
	(X,Y) = generateX(classOrReg, args.tfidf, tf, idf, locRates, N, VocabSize)

	if (classOrReg):
		clf = MultinomialNB()
		y = np.array(Y)
		K = 5
		k_fold = cross_validation.StratifiedKFold(Y, n_folds=K,shuffle=True, random_state=np.random.RandomState(seed))
		#TODO: should not be hard coded instead compute
		#fe_accuracies = [0]*len(50)
		for ja, topFE in enumerate(range(100, 5000, 100)):
			acc = 0
			acc_tf=0
			for k, (train, test) in enumerate(k_fold):
				ch2 = feature_selection.SelectKBest(feature_selection.chi2, k=topFE)
				X_new = ch2.fit_transform(X, y)
				Y_hat = clf.fit(X_new[train], y[train]).predict(X_new[test])
				cm = conf_mat(Y_hat, y[test])
				acc = (cm[0]+cm[2])/(len(Y_hat)) + acc
				
			#x_cord.append(topPercent)
			#fe_accuracies[ja] = acc/K
			print (topFE, acc/K)

	elif ( classOrReg ):		
		clf_v2 = linear_model.ElasticNetCV(l1_ratio=[0.75, 0.80, 0.85, 0.90, 0.95], n_jobs=3, cv=5, alphas=np.array([0.1, 1.0, 10, 100, 1000, 10000, 100000]))
		clf_v2.fit(X, Yreg)
		print (clf_v2.alpha_, clf_v2.l1_ratio_)
		print (clf_v2.mse_path_)



if __name__=="__main__":
	main()