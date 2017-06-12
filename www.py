#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'

import pandas as pd
import  numpy as np
import networkx as nx
import matplotlib.pyplot  as plt
from sklearn import preprocessing


dftrain2 = pd.read_table('./data/question_topic_train_set.txt', header=None, delimiter='\t',
                             names=['qid', 'topid'])
topicdf = pd.read_table('./data/topic_info.txt',header=None,delimiter='\t',
                      names = ['topid','ppid','tnc','tnw','tdc','tdw'])


topidId = topicdf['topid']
parentId = topicdf['ppid'].fillna('root')

edge = []

for i in xrange(len(topidId)):

    pidStr = parentId[i]
    # print pidStr
    pidList = pidStr.strip().split(',')

    for pl in pidList:
        edge.append((pidStr,pl))


G = nx.DiGraph()
G.add_edges_from(edge)

G.degree()      #度
# degreDict =  G.in_degree()#有向图，入度，return的是字典，
degdict = G.out_degree()
root = [k for k,v in degdict.items() if v == 0]

print 'root number is :',len(root)  #289
print 'trainset number is :',len(dftrain2) #2999967


enc = preprocessing.OneHotEncoder()
def process(dftrain2):

    for tp in dftrain2['topid'].fillna('empty'):

        tList = tp.strip().split(',')
        yield tList

enc.fit(process(dftrain2))

re = enc.transform(process(dftrain2)).toarray()

print re[0:10]
