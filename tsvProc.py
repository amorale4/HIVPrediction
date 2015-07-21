
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
#from operator import add;
seed=1003
random.seed(seed)
path2Data="cities/"

# reads the POS tags of the tweets in path2Data
# Since numbers and such give many unique ''words'' we can specify which to replace
# currently replacing '#'-hashtags, '@'-usernames, 'U'-url links, 'E'-emoticons, '$'- numeral , ','-punctuations, 'G'-unknown tag
# we are removing some formating symbols eg. ":" which is tagged to '~'

replaceables = ['#', '@', 'U', 'E', '$', ',', 'G']
#replaceables = []
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
    print ("sample size: ", sampleSize)
    
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
    print (tp, fp, tn, fn)
    return [tp, fp, tn, fn]


def main():
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
	
	N = 0
	for file in glob.glob(path2Data+'*.tsv'):
	    filenum = filenum + 1  #serves as an index for the file name
	    prefix = file.split('.')[0]
	    locRates[filenum] = int(prefix.split('_')[-1])
	    
	    lines = []
	    with open(file, 'r') as f:
	        lines = f.readlines()
	    #DEBUG
	    #print (file + " file num: " + str(filenum) + " num tweets: " + str(len(lines)) )
	    
	    unique_words = set([])
	    for line in lines:
	        ll = line.split('\t')
	        tweet = ll[0].strip()
	        tweet_pos = ll[1].strip()
	        
	        prevWord = "<s>"
	        for word in cleanTweet(tweet, tweet_pos):
	            if word not in V:
	                V[word]= Vcount
	                Vinv[Vcount]=word
	                Vcount = Vcount + 1
	            
	            #bigram 
	            if (prevWord,word) not in VV:
	            	VV[(prevWord, word)] = VVcount
	            	VVinv[VVcount] = (prevWord,word)
	            	VVcount = VVcount + 1

	            if V[word] not in idf:
	                idf[ V[word] ] = []
	            
	            if filenum not in tf:
	                tf[filenum] = {}
	            
	            freq = tf[filenum].get(V[word], 0)
	            tf[filenum][ V[word] ] = freq + 1
	            
	            if word not in unique_words:
	                idf[ V[word] ].append(filenum)
	                unique_words.add(word)

	            #bigram
	            if filenum not in bitf:
	                bitf[filenum] = {}
	            
	            bitf[filenum][VV[(prevWord, word)]] = bitf[filenum].get(VV[(prevWord,word)], 0) + 1
	            prevWord = word

	N = filenum
	VocabSize=len(V.keys())

	#some statistics about the data
	print ("vocabSize: ", VocabSize)
	print ("docSize: ", N)
	print ("labels: ", len(locRates))
	mu = mean(locRates.values())
	print ("mean: ",  mu )
	med = median(locRates.values())
	print ("median: ",  median(locRates.values()) )
	print ("max: ", max(locRates.values()) )
	print ("min: ", min(locRates.values()) )
	sigma = stdev(locRates.values())
	print ("standard diviation: ", sigma )

	#sample top/bottom rates
	topPercent = 0.10
	(validKeys, labels) = sampleItems(locRates, topPercent)
	
	# count1 = 0
	# count2 = 0
	#print ( idf[ V[ "#urbanradio" ]])
	# for key in idf [ V[ "hiv" ] ]:
	# 	tt = labels.get(locRates[key], -1)
	# 	if tt == 0:
	# 		count1 = count1 + 1
	# 	elif tt == 1:
	# 		count2 = count2 + 2
	# 	else:
	# 		print ("hiv occurs outside high/low examples:")
	# print ("number of Locations which are low counts and contain 'hiv': ", count1)
	# print ("number of Locations which are high counts and contain 'hiv': ", count2)
	# print ("out of: ", len(idf [ V[ "hiv" ] ]))

	#sample data
	p = 1.00 #percentage of sampled data (1.0) for cross-validation
	Yclass = []
	Yreg = []
	Ytest = []
	n = len(validKeys)
	validKeys0 = validKeys[:(int)(n/2)]
	validKeys1 = validKeys[(int)(n/2):]

	#need to keep it balanced
	dataSize = len(validKeys0)
	trainIndices0 = random.sample ( validKeys0, (int) (p*(dataSize)) ) #indicies start at 0
	testIndices0 = set(validKeys0) - set(trainIndices0)

	dataSize = len(validKeys1)
	trainIndices1 = random.sample ( validKeys1, (int) (p*(dataSize)) ) #indicies start at 1
	testIndices1 = set(validKeys1) - set(trainIndices1)

	testIndices = testIndices0.union(testIndices1)
	trainIndices = trainIndices0 + trainIndices1

	# generate matricies
	row = []
	col = []
	data = []
	for i, docID in enumerate(trainIndices):
	    for wordID in tf[docID]:
	        row.append(i)
	        col.append(wordID)
	        data.append(tfidf(docID, wordID, tf,idf, N) ) 

	    # bigram
	    for bigramID in bitf[docID]:
	    	row.append(i)
	    	col.append(VocabSize + bigramID)
	    	data.append(bitf[docID][bigramID])
	    
	    # uncomment to use regression
	    Yreg.append(locRates[docID]) 
	    # used for classification
	    Yclass.append (labels[ locRates[docID] ])
	
	#X = csr_matrix ( (np.array(data),(np.array(row),np.array(col))), shape=(len(trainIndices),VocabSize), dtype=float)
	#bigram
	X = csr_matrix ( (np.array(data),(np.array(row),np.array(col))), shape=(len(trainIndices),VocabSize+len(VV.keys())), dtype=float)
	print (X.shape)

	# #generate matricies (testing Data) Not used since we are doing cross-validation instead.
	# row = []
	# col = []
	# data = []
	# for i, docID in enumerate(testIndices):
	#     for wordID in tf[docID]:
	#         row.append(i)
	#         col.append(wordID)
	#         data.append(tfidf(docID, wordID, tf,idf) ) 
	#         # X[i, wordID] = tfidf(docID, wordID, tf,idf)
	#     #Ytest.append(locRates[docID])
	#     Ytest.append (labels[ locRates[docID] ])
	# Xtest = csr_matrix ( (np.array(data),(np.array(row),np.array(col))), shape=(len(testIndices),VocabSize), dtype=float)

	#print (sum(Ytrain), sum(Ytest))
	#print (X.shape)

	#classifier : http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
	# clf = pipeline.Pipeline([
	# 	('feature selection', feature_selection.SelectKBest(feature_selection.chi2, k=10))
	# 	('classification', MultinomialNB())
	# ])
	clf = MultinomialNB()
	#clf = linear_model.SGDClassifier()
	y = np.array(Yclass)
	y_ridge = np.array(Yreg)
	K = 5
	acc = 0
	prec = 0
	recal = 0
	k_fold = cross_validation.StratifiedKFold(Yclass, n_folds=K,shuffle=True, random_state=np.random.RandomState(seed))

	#clf_OLS = linear_model.LinearRegression()
	#clf_ridge = linear_model.Ridge()
	#clf_elastic = linear_model.ElasticNet()
	#avg_ols = 0
	#avg_rid = 0
	avg_ela = 0
	for k, (train, test) in enumerate(k_fold):
		#Y_hat = clf.fit(X[train], y[train]).predict(X[test])
		ch2 = feature_selection.SelectKBest(feature_selection.chi2, k=100)
		X_new = ch2.fit_transform(X, y)
		X_test = ch2.transform(X[test])
		
		#for feature in ch2.get_support(indices=True):
		#	print ( Vinv[feature] )
		Y_hat = clf.fit(X[train], y[train]).predict(X[test])

		cm = conf_mat(Y_hat, y[test])
		print ("confusion matrix:")
		print ("                             true label highHIVRates, true label lowHIVRates  ")
		print ("predicted label highHIVRates:  {0:{width}d} {1:{width}d}".format(cm[0], cm[1],width=15))
		print ("predicted label lowHIVRates :  {0:{width}d} {1:{width}d}".format(cm[3], cm[2],width=15))
		print ("[fold {0}], accuracy: {1:.5f}, precision: {2:.5f}, recall: {3:.5f}".
			format(k, (cm[0]+cm[2])/(len(Y_hat)), cm[0]/max(cm[0]+cm[1],1), cm[0]/max(cm[0]+cm[3],1)  ) )
		acc = (cm[0]+cm[2])/(len(Y_hat)) + acc
		prec = prec + cm[0]/max(cm[0]+cm[1],1)
		recal = recal + cm[0]/max(cm[0]+cm[3],1)
		
		print ("")
		#print (len(train))
		# t_rg = np.array(train)
		# print (t_rg)
		#clf_OLS.fit(X[train], y_ridge[train]) 
		#clf_ridge.fit(X[train], y_ridge[train]) 
		#clf_elastic.fit(X[train], y_ridge[train]) 
		
		#avg_ols = avg_ols + clf_OLS.score(X[test], y_ridge[test])
		#avg_rid = avg_rid + clf_ridge.score(X[test], y_ridge[test])
		#avg_ela = avg_ela + clf_elastic.score(X[test], y_ridge[test])
	

	print ("average acc: {0:.5f}, average precision: {1:.5f}, average recall: {2:.5f}".format(acc/K, prec/K, recal/K))
	#print ("avg OLS Regression: ",  avg_ols/K)
	#print ("avg Ridge Regression: ", avg_rid/K )
	#print ("avg Elastic Regression: ", avg_ela/K )

	# for regression use the following:

	#clf_ridge = linear_model.RidgeCV(cv=K)
	#clf_ridge.score(X,y)
	#print (clf_ridge.cv_values)
	
	# #1.0 is best possible score, lower is worse
	# clf.score(Xtest, Ytest)


	# # In[82]:

	# clf.fit (X, Ytrain) 
	# clf.coef_
	# #1.0 is best possible score, lower is worse
	# clf.score(Xtest, Ytest)


	# # In[ ]:

	# clf.fit (X, Ytrain) 
	# clf.coef_
	# #1.0 is best possible score, lower is worse
	# clf.score(Xtest, Ytest)


if __name__=="__main__":
	main()
