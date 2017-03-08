import time
import os
import pandas as pd

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
           