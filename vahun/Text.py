#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import numpy as np
import time
import pandas as pd

class Text:
    def __init__(self,
                corpus_path,
                size=0, #0 means all of them
                encoding_len=10, #max size
                printer=True,
                language="Hun",
                digraphs=False):
                  
        self.encoding_len=encoding_len

        self.full=[]
        self.printer=printer
        
        self.df = pd.read_csv(corpus_path, delimiter='\t',header=None)
        if size==0:
            size=len(self.df[0].values.tolist())
        # mix the words::
        self.wordlist=list(set(self.df[0].values.tolist()[0:size]))
        self.abc=self._getabc()
        
        self.all_features=self.featurize_data_charlevel_onehot(self.wordlist)
        train=self.all_features[0:int(len(self.all_features)*0.8)]
        valid=self.all_features[int(len(self.all_features)*0.8):int(len(self.all_features)*0.9)]
        test=self.all_features[int(len(self.all_features)*0.9):len(self.all_features)]
        self.x_train = train.reshape((len(train), np.prod(train.shape[1:])))
        self.x_valid= valid.reshape((len(valid), np.prod(valid.shape[1:])))
        self.x_test = test.reshape((len(test), np.prod(test.shape[1:])))
        
        
        
    def _getabc(self):
        abc=set()
        for word in self.wordlist:
            if type(word) is not float:
                for char in word:
                    abc.add(char)
        return "".join(sorted(abc, key=str.lower))
    
    def featurize_data_charlevel_onehot(self,x, maxlen=0):
        """
        @x: list of words
        @returns the feature tensor
        """
        if maxlen==0:
            maxlen=self.encoding_len
        self.feature_tensor = []
        for dix,item in enumerate(x):
            counter = 0
            one_hot = np.zeros((maxlen, len(self.abc)))
            if type(item) is not float:
                chars = list(item)
                if len(chars)<=maxlen:
                    for i in range(len(chars)):
                        if chars[i] in self.abc:
                            one_hot[maxlen-len(chars)+i,self.abc.find(chars[i])]=1
                    for i in range(maxlen-len(chars)):
                         one_hot[i,0]=1
                    self.feature_tensor.append(one_hot)
        self.feature_tensor=np.asarray(self.feature_tensor)
        return self.feature_tensor

    def defeaturize_data_charlevel_onehot(self,x,maxlen=0):
        """
        @x is the feature tensor
        @returns  the decoded word from the tensor
        """
        if maxlen==0:
            maxlen=self.encoding_len
        defeaturized=[]
        for item in x:
            out=""
            for i in range(maxlen):
                out+=self.abc[item[i,:].argmax()]
            defeaturized.append(out)
        return defeaturized
    

    

        