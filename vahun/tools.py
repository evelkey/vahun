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

class Timer:
    def __init__(self):
        self.timers=dict()
    def add(self,str):
        self.timers[str]=time.time()
    def get(self,str):
        return time.time()-self.timers[str]

class explog:
    def __init__(self,
                 feature_len,words,
                 unique_words,lang,
                 population_size,encoding_dim,
                 encoder_type,name="",sep="\t"):
        self.sep=sep
        self.dir="logs/"+name+"_"+time.strftime("%Y%m%d%H%M%S")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            self.logline("details.log",["Language",lang])
            self.logline("details.log",["Wordcount",words])
            self.logline("details.log",["Unique words",unique_words])
            self.logline("details.log",["Population size", population_size])
            self.logline("details.log",["Encoding dim",encoding_dim])
            self.logline("details.log",["Encoder type", encoder_type])    
        else:
            print("Folder already exists, expanding it.")
            
    def logline(self,file,array):
        string=""
        for item in array:
            string+=str(item)
            string+=self.sep
        string=string[0:len(string)-1]+"\n"
        with open(self.dir+'/'+file, "a") as myfile:
            myfile.write(string)
            
class logread:
    def __init__(self,logdir="logs",sep="\t"):
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
        for filename in os.listdir(os.getcwd()+"/"+self.logdir):
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
                    lenlist.append(len(self.fulltraineddata[i][j])-5)
                    trainlist.append(float(self.fulltraineddata[i][j][1]))
                    testlist.append(float(self.fulltraineddata[i][j][3]))
                    sumlist.append(sum(list(map(int, self.fulltraineddata[i][j][5:]))))
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
                    lenlist.append(len(self.accuracydata[i][j])-6)
                    trainlist.append(float(self.accuracydata[i][j][2]))
                    testlist.append(float(self.accuracydata[i][j][4]))
                    wordlist.append(float(self.accuracydata[i][j][5]))
                    sumlist.append(sum(list(map(int, self.accuracydata[i][j][6:]))))
                    if testlist[-1]>max_accuracy:
                        max_accuracy=testlist[-1]
                        best_config=list(map(int, self.accuracydata[i][j][6:]))
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
        wordlist=[]
        typelist=[]
        encodings=[]
        max_accuracy=0
        for i in range(len(self.detaildata)):
            if self.accuracydata[i]!=None:
                for j in range(len(self.accuracydata[i])):
                    names.append(self.detaildata[i][0])
                    encodings.append(float(self.detaildata[i][5][1]))
                    uniq.append("uniq" in self.detaildata[i][6][1])
                    typelist.append("var" in self.detaildata[i][6][1] or "var" in self.detaildata[i][0])
                    lenlist.append(len(self.accuracydata[i][j])-6)
                    trainlist.append(float(self.accuracydata[i][j][2]))
                    testlist.append(float(self.accuracydata[i][j][4]))
                    wordlist.append(float(self.accuracydata[i][j][5]))
                    configs.append(list(map(int, self.accuracydata[i][j][6:])))

        dt = [('Experiment', names),
         ('Encoded_len',encodings),
         ('Variational',typelist),
         ('Uniq',uniq),
         ('Layernum', lenlist),
         ('Train_char_acc.', trainlist),
         ('Test_char_acc.', testlist),
         ('Test_word_acc.', wordlist),
         ('Layers', configs)]
        return pd.DataFrame.from_items(dt)

    
def show_reconstruction(encoder,data,corp,length=0,inputdepth=10,inputfsize=38):
        enc_list=[]
        levenshteins=[]

        if isinstance(data,list):
            handmade=corp.featurize_data_charlevel_onehot(data)
            data=handmade.reshape((len(handmade), np.prod(handmade.shape[1:])))

        a=data
        b=(encoder.reconstruct(a))

        for i in range(len(data)):
            xa=corp.defeaturize_data_charlevel_onehot([a[i].reshape(inputdepth,inputfsize)])[0]
            xb=corp.defeaturize_data_charlevel_onehot([b[i].reshape(inputdepth,inputfsize)])[0]
            levenshteins.append(Levenshtein.distance(xa,xb))
            print(xa,"\t",xb,"\t",levenshteins[-1])

def show_performance(encoder,data,corp,length=0,plot=False,printer=False,inputdepth=10,inputfsize=38,encodim=180):
    enc_list=[]
    levenshteins=[]
    if length==0:
        length=len(data)
    if isinstance(data,list):
        handmade=corp.featurize_data_charlevel_onehot(data)
        data=handmade.reshape((len(handmade), np.prod(handmade.shape[1:])))
        if length==0:
            length=len(data)
    a=data
    b=(encoder.reconstruct(a))
    
    characc=np.ones(inputdepth)*length
    for i in range(len(data)):
        xa=corp.defeaturize_data_charlevel_onehot([a[i].reshape(inputdepth,inputfsize)])[0]
        xb=corp.defeaturize_data_charlevel_onehot([b[i].reshape(inputdepth,inputfsize)])[0]
        levenshteins.append(Levenshtein.distance(xa,xb))
        
        if i<length and printer:
            print(xa,"\t",xb,"\t",levenshteins[-1])
        for j in range(inputdepth):
            if (xa[j]!=xb[j]):
                characc[j]-=1
        enc_list.append(encoder.encode([a[i]])[0])
        if i<length and plot:
            plt.vlines([i for i in range(len(enc_list[-1]))],[0],enc_list[-1])
            plt.show()
    if printer:            
        print("\nAccuracy on data: ",encoder.char_accuracy(data)*100,"%")
    if plot:
        plt.plot([i for i in range(inputdepth)],characc/length)
        plt.show()
    
    #find high dev
    stds=[np.std([item[i] for item in enc_list]) for i in range(encodim)]
    if printer:
        print("average Levenshtein distance: ",np.mean(levenshteins))
    
    return stds