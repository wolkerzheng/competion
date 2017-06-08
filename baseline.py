#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
def loadVector():
    char2v = {}
    word2v = {}
    with open('./data/char_embedding.txt','r') as f:
        l = f.readline()
        # print f.readline()
        print l
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split(' ')
            char2v[tmp[0]]= np.array(map(float,tmp[1:]))
    with open('./data/word_embedding.txt','r') as f:
        l = f.readline()
        # print f.readline()
        print l
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split(' ')
            word2v[tmp[0]]= np.array(map(float,tmp[1:]))
    print char2v['c17']
    return char2v,word2v

def saveDict(word_centroid_map):

    with open('./data/word_centroid_map.txt','w') as f:
        for key,val in word_centroid_map.items():
            f.write('%s,%d\n'%(key,val))
    print 'save success'

def loadDict():
    word_centroid_map = {}
    with open('./data/word_centroid_map.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            # print line
            t = line.strip().split(',')
            # print t
            word_centroid_map[t[0]] = int(t[1])


    print 'load word_centroid_map success'
    return word_centroid_map


def getAverageFeatureForCharVec(charList,char2v):

    feature = np.zeros(256)
    count = 0
    keys = char2v.keys()
    charList = charList.strip().split(',')
    for c in charList:
        # print c
        if c in keys:
            feature = feature + char2v[c]
            count += 1
    if count!=0:
        feature = feature / count
    return feature

def getAverageFeatureForWordVec(wordList, word2v):

    feature = np.zeros(256)
    count = 0
    keys = word2v.keys()
    wordList = wordList.strip().split(',')
    for c in wordList:
        # print c
        if c in keys:
            feature = feature + word2v[c]
            count += 1
    if count != 0:
        feature = feature / count
    return feature

def getCosSimilarity(feature,topicMap):

    res = {}
    for key,val in topicMap.items():

        res[key] = np.dot(val,feature)
    sortedtopicidc = sorted(res.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in  sortedtopicidc[:5]]

def Main():

    startTime = time.clock()
    topicdf = pd.read_table('./data/topic_info.txt',header=None,delimiter='\t',
                      names = ['topid','ppid','tnc','tnw','tdc','tdw'])

    dftrain2 = pd.read_table('./data/question_topic_train_set.txt', header=None, delimiter='\t',
                             names=['qid', 'topid'])
    char2v, word2v = loadVector()
    #主题文件信息
    topicId = topicdf['topid']
    topicCharMatrix = np.zeros((256,len(topicdf['topid'])),dtype = 'float32')
    topicWordMatrix = np.zeros((256,len(topicdf['topid'])),dtype = 'float32')
    topicCharInfor = pd.Series(topicdf['tnc']+topicdf['tdc']).fillna('empty').tolist()
    topicWordInfor =  pd.Series(topicdf['tnw']+topicdf['tdw']).fillna('empty').tolist()
    for i in xrange(len(topicId)):
        topicCharMatrix[:,i] = getAverageFeatureForCharVec(topicCharInfor[i],char2v)
        topicWordMatrix[:,i] = getAverageFeatureForWordVec(topicWordInfor[i],word2v)

    print 'topicCharMatrix shape is :',topicCharMatrix.shape
    print 'topicWordMatrix shape is :', topicWordMatrix.shape
    # print char2v.keys()
    # num_cluster = 2000
    # print 'runing the char kmeans'
    # kmeans = KMeans(n_clusters=num_cluster)
    # idx = kmeans.fit_predict(char2v.values())
    #
    # print idx
    # word_centroid_map = dict(zip(char2v.keys(), idx))
    # print word_centroid_map[]
    # word_centroid_map = loadDict()
    print '...'
    # id =topicdf['topid']
    # cList = topicdf['tnc']+topicdf['tdc']
    # cList = cList.fillna('empty')
    # wordList = topicdf['tnw'] + topicdf['tdw']
    # wordList = wordList.fillna('empty')
    # topicMap = defaultdict(list)
    # for i in range(len(id)):
    #     # print cList[i]
    #     topicMap[id[i]].appendd(getAverageFeatureForCharVec(cList[i],char2v))
    #     topicMap[id[i]].appendd(getAverageFeatureForWordVec(wordList[i], word2v))
    # sub1 = pd.DataFrame(pd.Series(topicMap))
    # sub1.to_csv('topic.csv', index=False, sep=',', header=None)

    test = pd.read_table('./data/question_eval_set.txt', header=None, delimiter='\t',
                         names=['qid', 'tc', 'tw', 'dc', 'dw'])

    result = []
    testId = test['qid']
    print '训练集大小为：',len(testId)
    testCharMatrix = np.zeros((len(test['qid']),256),dtype = 'float32')
    testWordMatrix = np.zeros((len(test['qid']),256),dtype = 'float32')
    testCharInfor = pd.Series(test['tc'] + test['dc']).fillna('empty').tolist()
    testWordInfor = pd.Series(test['tw'] + test['dw']).fillna('empty').tolist()
    print 'compute cos'
    for i in range(len(testId)):
        if i%1000 == 0:
            print 'process %d the test'%(i+1)
        # print cList[i]
        testCharMatrix[i,:] = getAverageFeatureForCharVec(testCharInfor[i], char2v)
        testWordMatrix[i,:] = getAverageFeatureForWordVec(testWordInfor[i], word2v)

    charSilimaity = np.dot(testCharMatrix,topicCharInfor)
    wordSilimaity = np.dot(testWordMatrix, topicWordInfor)
    print charSilimaity.max(axis=1)
    print 'writing ... '
    # print result
    # # res = []
    # for key,val in result.items():
    #     print key,val
    # sub = pd.DataFrame(res)
    # sub.to_csv('simple_cluster.csv', index=False, sep=',', header=None)
    sub = pd.DataFrame(result)
    sub.to_csv('simple_cluster.csv', index=False, sep=',', header=None)
    ##write_to_txt
    endTime = time.clock()

    print 'total running time is :',endTime-startTim
            #十个聚类
    # for cluster in xrange(0, 10):
    #     #
    #     # Print the cluster number
    #     print "\nCluster %d" % cluster
    #     #
    #     # Find all of the words for that cluster number, and print them out
    #     words = []
    #     for i in xrange(0, len(word_centroid_map.values())):
    #         if (word_centroid_map.values()[i] == cluster):
    #             words.append(word_centroid_map.keys()[i])
    #     print words

# loadDict()
Main()
