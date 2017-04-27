#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import heapq
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from vahun.neuroplot import DrawNN
import Levenshtein

class LogReader:
    def __init__(self,logdir="/mnt/store/velkey/logs",sep="\t"):
        self.logdir=logdir
        self.delimiter=sep
        self.lognames=self.read_all_log_folders()
        self.detaildata=[]
        self.traindata=[]
        self.fulltraineddata=[]
        self.accuracydata=[]
        for folder in self.lognames:
            if os.path.exists(self.logdir+"/"+folder+"/"+"details.log"):
                self.detaildata.append([folder]+self.read_linebyline(folder,"details.log"))
            else:
                self.detaildata.append(None)
                self.traindata.append(self.read_train(folder))
            if os.path.exists(self.logdir+"/"+folder+"/"+"full_trained.log"):
                self.fulltraineddata.append(self.read_linebyline(folder,"full_trained.log"))
            else:
                self.fulltraineddata.append(None)
            if os.path.exists(self.logdir+"/"+folder+"/"+"accuracy.log"):
                self.accuracydata.append(self.read_linebyline(folder,"accuracy.log"))
            else:
                self.accuracydata.append(None)
            
    def read_all_log_folders(self):
        lista=[]
        for filename in os.listdir(self.logdir):
            lista.append(filename)
        return lista
    def read_train(self,folder):
        train_logs=[]
        if os.path.exists(self.logdir+"/"+folder+"/"+"train.log"):
            with open(self.logdir+"/"+folder+"/"+"train.log") as f:
                for line in f:
                    if "START" in line:
                        temptrain=[]
                    elif "STOP" in line:
                        train_logs.append(temptrain)
                    else: 
                        temptrain.append(line.rstrip('\n').split(self.delimiter))
        return train_logs
    def read_linebyline(self,folder,logname):
        logs=[]
        if os.path.exists(self.logdir+"/"+folder+"/"+logname):
            with open(self.logdir+"/"+folder+"/"+logname) as f:
                for line in f:
                    logs.append(line.rstrip('\n').split(self.delimiter))
        else:
            print("No logfile found")
        return logs
    
    def plot_fulltrains(self):
        for i in range(len(self.detaildata)):
            print(self.detaildata[i])
            lenlist=[]
            trainlist=[]
            testlist=[]
            sumlist=[]
            if self.fulltraineddata[i]!=None:
                for j in range(len(self.fulltraineddata[i])):
                    lenlist.append(len(self.fulltraineddata[i][j])-12)
                    trainlist.append(float(self.fulltraineddata[i][j][3]))
                    testlist.append(float(self.fulltraineddata[i][j][7]))
                    sumlist.append(sum(list(map(int, self.fulltraineddata[i][j][12:]))))
                plt.plot(lenlist,trainlist,'ro',lenlist,testlist,'g^')

                plt.show()
    def plot_accuracy(self,filter="",plot=True,draw=True):
        for i in range(len(self.detaildata)):
            
            lenlist=[]
            trainlist=[]
            testlist=[]
            sumlist=[]
            wordlist=[]
            max_accuracy=0
            if self.accuracydata[i]!=None and filter in self.detaildata[i][6][1]:
                print(self.detaildata[i])
                for j in range(len(self.accuracydata[i])):
                    lenlist.append(len(self.accuracydata[i][j])-12)
                    trainlist.append(float(self.accuracydata[i][j][3]))
                    testlist.append(float(self.accuracydata[i][j][7]))
                    wordlist.append(float(self.accuracydata[i][j][8]))
                    sumlist.append(sum(list(map(int, self.accuracydata[i][j][12:]))))
                    if testlist[-1]>max_accuracy:
                        max_accuracy=testlist[-1]
                        best_config=list(map(int, self.accuracydata[i][j][12:]))
                if plot:
                    plt.plot(sumlist,trainlist,'ro',sumlist,testlist,'g^',sumlist,wordlist,'bx')
                print("Max test accuracy:",max_accuracy,
                     "\nMax train accuracy:",max(trainlist),
                     "\nWith config: ",best_config)
                if plot:
                    plt.show()
                if draw:
                    DrawNN([int(best_config[-1]/10)]+[int(item/10) for item in best_config]).draw()
    def get_full_table(self):
        lenlist=[]
        uniq=[]
        names=[]
        trainlist=[]
        testlist=[]
        configs=[]
        testwordlist=[]
        wordnum=[]
        typelist=[]
        encodings=[]
        validlist=[]
        levenshteins_tr=[]
        levenshteins_te=[]
        levenshteins_v=[]
        max_accuracy=0
        for i in range(len(self.detaildata)):
            if self.accuracydata[i]!=None:
                for j in range(len(self.accuracydata[i])):
                    names.append(self.detaildata[i][0])
                    wordnum.append(self.detaildata[i][3][1])
                    
                    uniq.append("uniq" in self.detaildata[i][6][1] or "uniq" in self.detaildata[i][0])
                    typelist.append("var" in self.detaildata[i][6][1] or "var" in self.detaildata[i][0])
                    lenlist.append(len(self.accuracydata[i][j])-12)
                    trainlist.append(float(self.accuracydata[i][j][3]))
                    validlist.append(float(self.accuracydata[i][j][5]))
                    testlist.append(float(self.accuracydata[i][j][7]))
                    testwordlist.append(float(self.accuracydata[i][j][8]))
                    levenshteins_tr.append(float(self.accuracydata[i][j][9]))
                    levenshteins_v.append(float(self.accuracydata[i][j][10]))
                    levenshteins_te.append(float(self.accuracydata[i][j][11]))
                    configs.append(list(map(int, self.accuracydata[i][j][12:])))
                    encodings.append(min(configs[-1]))

        dt = [('Experiment', names),
         ('Encoded_len',encodings),
         ('Uniq_words', wordnum),
         ('Variational',typelist),
         ('Uniq',uniq),
         ('Layernum', lenlist),
         ('Train_char_acc', trainlist),
         ('Valid_char_acc', validlist),
         ('Test_char_acc', testlist),
         ('Test_word_acc', testwordlist),
         ('Test_Leven_avg', levenshteins_te),
         ('Train_Leven_avg', levenshteins_tr),
         ('Valid_Leven_avg', levenshteins_v),
         ('Layers', configs)]
 
        return pd.DataFrame.from_items(dt)