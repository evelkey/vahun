import sys
import tensorflow as tf
from vahun.Text import Text
import numpy as np
from vahun.tools import Timer
from vahun.tools import explog
from vahun.autoencoder import Autoencoder_ffnn
from vahun.variational_autoencoder import Variational_autoencoder
from vahun.genetic import evolution
from vahun.genetic import experiment
from vahun.tools import show_performance
from vahun.genetic import Settings
import argparse
import random


def levennoise(corpus,word,dist=2):
    if len(word)>18:
        return word
    for i in range(dist):
        a=random.random()
        if a <= 0.333:
            #del
            r=random.randint(0,len(word))
            word=word[:r]+word[r+1:]
        if a>0.333 and a<=0.666:
            #append
            r=random.randint(0,len(word))
            x=random.randint(0,len(corpus.abc))-1
            ch=list(corpus.abc)[x]
            word=word[:r]+ch+word[r:]
        if a>0.666:
            #change
            r=random.randint(0,len(word))
            x=random.randint(0,len(corpus.abc))-1
            ch=list(corpus.abc)[x]
            word=word[:r]+ch+word[r+1:]
    return word

def levenshtein_noisify(corpus):
    wordlist=[levennoise(corpus,word) for word in corpus.wordlist]
    return wordlist,corpus.featurize_data_charlevel_onehot(wordlist)

def main(args=None):
    timer=Timer()

    size=args.corp_len


    corpuses=[Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20',size=size),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20',size=size),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20.digraph_repl',size=size),
              Text(corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20.digraph_repl',size=size)]
    print("Corpus list ready")
    
    exps = []
    ranger=range(args.low,args.high)
    i=0
    with open('/mnt/store/velkey/levenshteins') as f:
        for line in f:
            if(i in ranger):
                exps.append(line.strip().split('\t'))
            i+=1


    for exper in exps:
        exper=[int(item) for item in exper]
        layerlist=exper[3:]
        settings=Settings(layerlist)
        typ=0
        if exper[1]==0 and exper[2]==0:
            corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20'
            typ=0
        if exper[1]==1 and exper[2]==0:
            corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20'
            typ=1
        if exper[1]==0 and exper[2]==1:
            corpus_path='/mnt/store/velkey/mnsz2/filt.200k.maxlen20.digraph_repl'
            typ=2
        if exper[1]==1 and exper[2]==1:
            corpus_path='/mnt/store/velkey/mnsz2/filt.200k_random.maxlen20.digraph_repl'
            typ=3

        corpus=corpuses[typ]
        words,X_levenshtein=levenshtein_noisify(corpus)
        name=(str("uniq_levenshtein_"+("variational_" if exper[0]==1 else "autoencoder_")+
                        ("top_" if exper[1]==0 else "random_")+
                        ("bigraph_" if exper[2]==1 else "uni_")))
        logger=explog(encoder_type=name,
                  encoding_dim=min(settings.weights),
                  feature_len=20,
                  lang=corpus_path,
                  unique_words=len(set(corpus.wordlist)),
                  name=name,
                  population_size=0,
                  words=len(corpus.wordlist))
        for k in range(2):
            print("starting experiment: ",exper)
            timer.add("experiment")

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            if(exper[0]==1):
                encoder=Variational_autoencoder(logger=logger,
                                                tf_session=sess,
                                                inputdim=len(corpus.abc)*20,
                                                encoding_size=settings.weights[0],
                                                corpus=corpus,
                                                optimizer =tf.train.AdamOptimizer(learning_rate = 0.001),
                                                nonlinear=tf.sigmoid,charnum=len(corpus.abc))


            else:
                encoder=Autoencoder_ffnn(
                                 experiment=settings,
                                 logger = logger,
                                 tf_session=sess,
                                 inputdim = len(corpus.abc)*20,
                                 layerlist = settings.weights,
                                 encode_index = int(len(settings.weights)/2),
                                 corpus = corpus,
                                 optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                 nonlinear = tf.sigmoid,
                                 charnum=len(corpus.abc))

            encoder.train(X_train=X_levenshtein,corpus.x_valid,corpus.x_test,512,80,Y_train=corpus.x_train)

            print("Finished in:", timer.get("experiment") ,"s")
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder experiment')
    parser.add_argument("--corp_len", dest="corp_len", default=0, type=int, help="Words to read from corpus")
    parser.add_argument("--feature_len", dest="feature_len", default=20, type=int, help="Feature size")
    parser.add_argument("--from", dest="low", type=int, help="lower boundary")
    parser.add_argument("--to", dest="high",  type=int, help="upper boundary")

    args = parser.parse_args()

    main(args)