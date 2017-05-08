#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import numpy as np
import time
import pandas as pd

class Corpus:
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
    
class TSV_Corpus(Corpus):
    def __init__(self,
                corpus_path,col=0,
                size=0, #0 means all of them
                encoding_len=20, #max size
                printer=True,
                language="Hun"):
                  
        self.encoding_len=encoding_len


        self.printer=printer
        
        self.df = pd.read_csv(corpus_path, delimiter='\t',header=None)
        if size==0:
            size=len(self.df[col].values.tolist())
        # mix the words::
        adf=self.df.values.tolist()
        self.wordlist=[adf[i][col] for i in range(0,size)]
        self.abc=" " + self._getabc()
        
        self.all_features=self.featurize_data_charlevel_onehot(self.wordlist)
        train=self.all_features[0:int(len(self.all_features)*0.8)]
        valid=self.all_features[int(len(self.all_features)*0.8):int(len(self.all_features)*0.9)]
        test=self.all_features[int(len(self.all_features)*0.9):len(self.all_features)]
        self.x_train = train.reshape((len(train), np.prod(train.shape[1:])))
        self.x_valid= valid.reshape((len(valid), np.prod(valid.shape[1:])))
        self.x_test = test.reshape((len(test), np.prod(test.shape[1:])))
        

        self.df=None
        self.all_features=None
        
class WPL_Corpus:
    def __init__(self,
                corpus_path,
                size=4000000,
                language="Hun",
                needed_corpus=["unique","lower","hun_lower","lower_unique","hun_lower_unique"],
                encoding_len=10,
                corpus_stream=None,
                printer=True,
                new=True):
        """
        Creates corpus object, with the given parameters
                @needed_corpus: list, can contain: "unique","lower","hun_lower","lower_unique","hun_lower_unique"
        """
                  
        self.encoding_len=encoding_len
        self.symbol='0123456789-,;.!?:’\”\"/\\|_@#$%^&*~`+ =<>()[]{}'
        self.accents = 'áéíóöőúüű'
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.space = ' '
        self.beginend= '^$'
        
        if language=="Hun":
            self.abc=self.space+self.alphabet+self.accents+self.beginend
        
        self.language=language
        
        self.embedder=np.random.normal(size=[len(self.abc),encoding_len])
        
        self.full=[]
        self.printer=printer
        
        if corpus_stream==None:
            self.corpus_path=corpus_path
            self.read_all_words(size)
        else:
            self.corpus_path=None
            self.read_stream(corpus_stream,size)
        
        if "unique" in needed_corpus:
            self.unique=list(set(self.full))
        if "lower" in needed_corpus:
            self.lower=self.lowercasen(self.full)
        if "lower_unique" in needed_corpus:
            self.lower_unique=list(set(self.lowercasen(self.full)))
        if "hun_lower" in needed_corpus:  
            self.hun_lower=self.filter_only_words(self.lowercasen(self.full))
        if "hun_lower_unique" in needed_corpus:
            self.hun_lower_unique=list(set(self.filter_only_words(self.lowercasen(self.full))))
            
        if self.printer: print("Corpus initalized, fields:",needed_corpus,"\nUnique words: ",len(set(self.full)))
        
        
    def is_hun_word(self,word):
        """
        @word: char sequence without spaces
        @return: true if the word can be hungarian, no symbols included
        """
        hun_word=True
        if "eeeoddd" in word or ' ' in word or ""==word:
            return False
        for char in self.symbol:
            if char in word:
                return False
        return hun_word
    
    def read_line_wpl(self,line):
        """
        Reads a line from a world per line format
        @line: line in file
        @return: the word
        """
        return line.replace("\n","")
    
    def lowercasen(self,list_of_words):
        return [word.lower() for word in list_of_words]
    
    def filter_only_words(self,corpus):
        if self.language != "Hun":
            return []
        return [word for word in corpus if self.is_hun_word(word)]
            
    def read_all_words(self,size,format="wpl"):
        """
        Reads words from the specified format
        @size: max number of words
        """
        i=0
        start=time.time()
        
        with open(self.corpus_path,encoding='utf8') as f:
            for line in f:
                if i==size:
                    break
                else:
                    if format=="wpl" :
                        self.full.append(self.read_line_wpl(line))
                    i+=1
                    if i%1000000==0: 
                        if i!=0 and self.printer: print("Reading file, speed: ",1000000/(time.time()-start)," words/s")
                        start=time.time()
                        
    def read_stream(self,stream,size,format="wpl"):
        """
        Reads words from the specified format
        @size: max number of words
        """
        i=0
        start=time.time()
        
        with stream as f:
            for line in f:
                if i==size:
                    break
                else:
                    if format=="wpl" :
                        self.full.append(self.read_line_wpl(line))
                    i+=1
                    if i%1000000==0: 
                        if i!=0 and self.printer: print("Reading file, speed: ",1000000/(time.time()-start)," words/s")
                        start=time.time()              
    

    def get_stat(self,corpus):
        frequency=collections.Counter(corpus)
        return frequency
    
    def create_most_common_corpus(self,corpus,count):
        self.most_common=[]
        for item in self.get_stat(corpus).most_common(count):
            self.most_common.append(item[0])
        return self.most_common
    
    def get_random_block_from_data(self,batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return self.feature_tensor[start_index:(start_index + batch_size)]
    
    def mark_begin_end(self,word):
        return "^" + word + "$"
    
    def mark_list(self,lista):
        return [self.mark_begin_end(word) for word in lista]
        
    def di2single(self,word):
        word=word.replace("cs","C")
        word=word.replace("ly","J")
        word=word.replace("zs","Z")
        word=word.replace("ny","N")
        word=word.replace("dz","D")
        word=word.replace("dzs","K")
        word=word.replace("sz","S")
        word=word.replace("ty","T")
        word=word.replace("gy","G")
        return word
    
    def digraph_2_single(self,lista):
        return [self.di2single(word) for word in lista]
        

        