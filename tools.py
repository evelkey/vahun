import time

class Timer:
    def __init__(self):
        self.timers=dict()
    def add(self,str):
        self.timers[str]=time.time()
    def get(self,str):
        return time.time()-self.timers[str]


def logtsv(array):
    string=""
    for item in array:
        string+=str(item)
        string+="\t"
    string=string[0:len(string)-1]+"\n"
    with open("train.tsv", "a") as myfile:
        myfile.write(string)